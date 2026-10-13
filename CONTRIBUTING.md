# Contributing

This repository is a small research snapshot. Contributions that correct a claim,
explain a negative result, or reproduce a bounded experiment are welcome. No model
weights, inference server, remote deployment or safety controller are included.

For host-side changes, run the commands in [VALIDATION.md](VALIDATION.md), preserve
the tests, and describe the problem and validation result. Refresh the manifest
with `python3 tools/validate_release.py --write-manifest` after reviewing edits.
A normal validator run checks hashes without changing them.

For kernel changes, document the target GPU/toolchain, shape, data layout, input
provenance, numerical acceptance rule, memory requirement and bounded run plan
before timing. Compare identical inputs, check the normalized output against an
independent oracle, check empty/masked rows and memory accesses, and retain failed
results. A speed claim needs raw timings and an identified source revision.
Do not loosen a tolerance after seeing a failure without explaining and reviewing
that change as a new numerical contract.

Keep historical records identifiable. Add a new result rather than replacing an
old experiment. Avoid private paths, machine addresses, credentials, model weights
or copyrighted text in contributions. Cite the origin and license of imported
code; linked projects are not automatically covered by this repository's MIT license.

Never use this example to bypass an existing resource controller, reservation,
cleanup blocker or budget. The example's free-memory preflight is not a system
safety framework. Report the exact error and environment when a platform has not
been validated; do not label unexecuted commands as passing tests.
