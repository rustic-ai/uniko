# Security Policy

We take the security of uniko seriously and appreciate responsible disclosure
of vulnerabilities.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, use one of the following private channels:

1. **Preferred — GitHub private security advisories.** Open a private report at
   <https://github.com/rustic-ai/uniko/security/advisories/new>. This keeps the
   details confidential while we investigate and coordinate a fix.
2. **Email.** If you cannot use GitHub advisories, email
   **security@dragonscale.ai**.

### What to include

To help us triage and reproduce the issue quickly, please include as much of the
following as you can:

- A description of the vulnerability and its potential impact.
- The affected crate(s) and version(s) or commit hash.
- Step-by-step instructions to reproduce, including a minimal proof of concept
  where possible.
- Any relevant logs, stack traces, or configuration (with secrets redacted).
- Your assessment of severity and any suggested remediation.

Please give us a reasonable opportunity to investigate and address the issue
before any public disclosure.

## Response Timeline

- **Acknowledgement:** we aim to acknowledge your report within a few business
  days.
- **Assessment:** we will work to confirm the issue, determine its impact, and
  keep you informed of our progress.
- **Resolution:** once a fix is available, we will coordinate disclosure and,
  where appropriate, credit the reporter.

## Supported Versions

uniko is in early, active development. Security fixes are provided for the
current `0.6.x` release series.

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| < 0.6   | :x:                |

As the project matures, this policy will be updated to reflect a longer-term
support window.

## Contact

- Security reports: **security@dragonscale.ai**
- General inquiries: **dev@dragonscale.ai**
