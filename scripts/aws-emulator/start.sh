#!/usr/bin/env bash
# Start the AWS Emulator for the object-store lease tests, then print the
# environment those tests read.
#
#   set -a; eval "$(scripts/aws-emulator/start.sh)"; set +a
#   cargo nextest run -p uniko-store --lib -E 'test(emulator)'
#
# Builds the image from ./Dockerfile (AWS publishes none), runs it with
# authentication and telemetry off, waits for its health check, and creates
# the test bucket. The emulator writes a fresh self-signed CA at each start;
# the tests trust it through UNIKO_S3_EMULATOR_CA.
#
# Env: CONTAINER (docker or podman; default: whichever is on PATH, docker
# first), PORT (default 9297), BUCKET (default uniko-lease-tests),
# EMULATOR_HOME (default target/aws-emulator-home).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
container="${CONTAINER:-$(command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && echo docker || echo podman)}"
port="${PORT:-9297}"
bucket="${BUCKET:-uniko-lease-tests}"
home="${EMULATOR_HOME:-$root/target/aws-emulator-home}"
case "$(uname -m)" in
  arm64 | aarch64) arch=aarch64 ;;
  *) arch=x86_64 ;;
esac

mkdir -p "$home"
# The image runs as uid 1000; the host user may not be (CI runners are 1001).
chmod a+rwx "$home"
"$container" build --quiet --build-arg ARCH="$arch" -t aws-emulator "$here" >&2
"$container" rm -f uniko-aws-emulator >/dev/null 2>&1 || true
"$container" run -d --name uniko-aws-emulator --memory=2g \
  -p "127.0.0.1:$port:$port" -e HOME="$home" -v "$home:$home" \
  aws-emulator -p "$port" -a off -t off >&2

ca="$home/.aws/emulator/$port/ca.pem"
endpoint="https://127.0.0.1:$port"
for _ in $(seq 1 60); do
  if [ -f "$ca" ] && curl -sf --cacert "$ca" "$endpoint/__admin/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -sf --cacert "$ca" "$endpoint/__admin/health" >/dev/null || {
  echo "the emulator did not become healthy" >&2
  "$container" logs uniko-aws-emulator >&2 || true
  exit 1
}

# Every service shares the port: the SigV4 scope tells it this is S3. With
# authentication off any credentials sign.
curl -sf -o /dev/null --cacert "$ca" --aws-sigv4 "aws:amz:us-east-1:s3" \
  --user "test:test" -X PUT "$endpoint/$bucket" >&2 || true

# `KEY=VALUE` lines: append them to $GITHUB_ENV in CI, or `export` them
# locally: `set -a; eval "$(scripts/aws-emulator/start.sh)"; set +a`.
echo "UNIKO_S3_EMULATOR_ENDPOINT=$endpoint"
echo "UNIKO_S3_EMULATOR_CA=$ca"
echo "UNIKO_S3_EMULATOR_BUCKET=$bucket"
