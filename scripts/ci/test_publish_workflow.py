"""Exercise the release workflow's actual Bash with fake network and Cargo tools.

Run with: python3 -m unittest discover -s scripts/ci -p 'test_*.py'
No credentials, registry requests, builds, or uploads are performed.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


REPO = Path(__file__).resolve().parents[2]


def publish_script():
    workflow = (REPO / ".github/workflows/release.yml").read_text()
    step = workflow.split("      - name: Publish in dependency order\n", 1)[1]
    run = step.split("        run: |\n", 1)[1]
    lines = []
    for line in run.splitlines():
        if line.strip() and not line.startswith("          "):
            break
        lines.append(line)
    return textwrap.dedent("\n".join(lines))


FAKE_TOOL = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
log = Path(os.environ["PUBLISH_TEST_LOG"])
previous = [json.loads(line) for line in log.read_text().splitlines()]
with log.open("a") as out:
    out.write(json.dumps({"tool": tool, "args": args}) + "\n")
scenario = os.environ["PUBLISH_TEST_SCENARIO"]
if tool == "cargo":
    assert "--locked" in args
    if "--dry-run" in args:
        assert not os.environ.get("CARGO_REGISTRY_TOKEN")
        if scenario == "already_uploaded":
            print("error: crate already uploaded")
            sys.exit(101)
        if scenario == "verification_failure":
            print("error: package verification failed")
            sys.exit(101)
        print("package verified")
    else:
        assert "--no-verify" in args
        assert os.environ["CARGO_REGISTRY_TOKEN"] == "test-upload-token"
        if scenario == "upload_auth_failure":
            print("error: HTTP 403 Invalid authentication token")
            sys.exit(101)
        uploads = sum(p["tool"] == "cargo" and "--no-verify" in p["args"] for p in previous)
        if scenario == "always_rate_limited" or (scenario == "rate_limited_once" and uploads == 0):
            print("error: HTTP 429 too many requests")
            sys.exit(101)
        print("published")
elif tool == "curl":
    method = args[args.index("-X") + 1] if "-X" in args else "GET"
    if method == "GET":
        if scenario == "oidc_failure":
            print('{"message":"OIDC request denied"}')
            sys.exit(22)
        print('{"value":"test-jwt"}')
    else:
        assert args[args.index("-A") + 1] == "uniko-release (https://github.com/rustic-ai/uniko)"
        if method == "POST":
            assert "--fail-with-body" in args
            assert json.load(sys.stdin) == {"jwt": "test-jwt"}
            # Verification must precede the exchange, not merely the upload.
            assert previous[-1]["tool"] == "curl"
            assert any(p["tool"] == "cargo" and "--dry-run" in p["args"] for p in previous)
            if scenario == "exchange_failure":
                print('{"errors":[{"detail":"trusted publisher mismatch"}]}')
                sys.exit(22)
            if scenario == "missing_token":
                print('{}')
            else:
                print('{"token":"test-upload-token"}')
        elif method == "DELETE":
            assert "Bearer test-upload-token" in args
elif tool == "df":
    print("filesystem has space")
'''


@unittest.skipUnless(shutil.which("jq"), "workflow requires jq")
class PublishWorkflowTests(unittest.TestCase):
    def run_workflow(self, scenario):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "calls.jsonl"
            log.touch()
            for name in ("cargo", "curl", "sleep", "df"):
                tool = root / name
                tool.write_text(FAKE_TOOL)
                tool.chmod(0o755)
            env = os.environ.copy()
            env.pop("CARGO_REGISTRY_TOKEN", None)
            env.update(
                PATH=f"{root}{os.pathsep}{env['PATH']}",
                PUBLISH_TEST_LOG=str(log),
                PUBLISH_TEST_SCENARIO=scenario,
                PUBLISH_ORDER="uniko-store uniko-pipes",
                ACTIONS_ID_TOKEN_REQUEST_TOKEN="test-request-token",
                ACTIONS_ID_TOKEN_REQUEST_URL="https://example.invalid/oidc?test=1",
            )
            result = subprocess.run(
                ["bash", "-e", "-c", publish_script()],
                env=env, capture_output=True, text=True, timeout=30,
            )
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return result, calls

    def test_verifies_before_minting_and_revokes_each_upload_token(self):
        result, calls = self.run_workflow("success")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        meaningful = [call for call in calls if call["tool"] in ("cargo", "curl")]
        sequence = [
            "verify" if "--dry-run" in call["args"] else "upload"
            if call["tool"] == "cargo" else call["args"][call["args"].index("-X") + 1]
            if "-X" in call["args"] else "GET"
            for call in meaningful
        ]
        self.assertEqual(sequence, ["verify", "GET", "POST", "upload", "DELETE"] * 2)
        self.assertIn("::add-mask::test-jwt", result.stdout)
        self.assertIn("::add-mask::test-upload-token", result.stdout)

    def test_token_exchange_rejection_reports_body_and_stops(self):
        result, calls = self.run_workflow("exchange_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("trusted publisher mismatch", result.stdout)
        self.assertIn("environment release", result.stdout)
        self.assertEqual(sum(call["tool"] == "curl" for call in calls), 2)
        self.assertFalse(any("--no-verify" in call["args"] for call in calls))

    def test_oidc_rejection_identifies_github_and_stops(self):
        result, calls = self.run_workflow("oidc_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GitHub OIDC request failed", result.stdout)
        self.assertIn("OIDC request denied", result.stdout)
        self.assertEqual(sum(call["tool"] == "curl" for call in calls), 1)

    def test_missing_token_never_uploads(self):
        result, calls = self.run_workflow("missing_token")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not contain a publishing token", result.stdout)
        self.assertFalse(any("--no-verify" in call["args"] for call in calls))

    def test_upload_authorization_failure_revokes_and_stops(self):
        result, calls = self.run_workflow("upload_auth_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("publishing authorization failed", result.stdout)
        self.assertEqual(sum("--no-verify" in call["args"] for call in calls), 1)
        self.assertEqual(sum("DELETE" in call["args"] for call in calls), 1)

    def test_failed_verification_never_requests_token(self):
        result, calls = self.run_workflow("verification_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("package verification failed", result.stdout)
        self.assertFalse(any(call["tool"] == "curl" for call in calls))

    def test_already_uploaded_crates_are_skipped(self):
        result, calls = self.run_workflow("already_uploaded")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any(call["tool"] == "curl" for call in calls))

    def test_rate_limit_retries_same_crate_with_a_fresh_token(self):
        result, calls = self.run_workflow("rate_limited_once")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        uploads = [call for call in calls if "--no-verify" in call["args"]]
        self.assertEqual([call["args"][2] for call in uploads], ["uniko-store", "uniko-store", "uniko-pipes"])
        self.assertEqual(sum("DELETE" in call["args"] for call in calls), 3)

    def test_rate_limit_eventually_fails_instead_of_retrying_forever(self):
        result, calls = self.run_workflow("always_rate_limited")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("after 20 attempts", result.stdout)
        self.assertEqual(sum("--no-verify" in call["args"] for call in calls), 20)
        self.assertEqual(sum("DELETE" in call["args"] for call in calls), 20)


if __name__ == "__main__":
    unittest.main()
