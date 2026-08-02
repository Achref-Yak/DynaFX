# Security Policy

## Supported Versions

DynaFX is under active development (0.x). We currently support the latest
release on `main`.

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |

## Reporting a Vulnerability

Security issues should be reported privately — **not** via the public issue
tracker.

**How to report:**

1. Open a private advisory through GitHub's
   [Report a vulnerability](https://github.com/Achref-Yak/DynaFX/security/advisories/new)
   flow on the repository.
2. Include the affected version, a minimal reproduction, and the impact.

**What happens next:**

- You'll receive an acknowledgment within 5 business days.
- The maintainer will assess the report and, if confirmed, work toward a fix
  and a release note.
- Please avoid publicly disclosing the issue until it is addressed.

## Scope

DynaFX is a research/simulation library. Its threat model is primarily about
**malicious `.sysd` models, Turtle/RDF data, or Python extensions** executed in
a trusted research environment — treat any untrusted model or data source as
arbitrary code. Bugs that cause crashes or incorrect simulation results on
trusted inputs are still appreciated, and can be filed as regular issues.
