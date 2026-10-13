# v0.1 validation status

The release is a **research source snapshot**. It contains an implemented CUDA
example and historical measurements. It is not a newly GPU-qualified standalone
binary or an end-to-end offload system.

| Check | Status for the final package |
|---|---|
| CMake CPU-only configure/build and CTest | Passed locally on macOS / AppleClang 21 |
| Host shape/API checks | 98 passed |
| Strict CLI boundary checks | 15 passed |
| Independent scalar oracle host checks | 6,149 passed |
| Host address/undefined-behavior sanitizer | Passed locally |
| Local links, JSON, manifest and historical arithmetic | Checked by the release validator |
| NVCC compilation/linking of this standalone extraction | **Not run** |
| GPU self-test, numerical acceptance and repeated-output checks | **Not run** |
| GPU memory checking / compute-sanitizer | **Not run** |
| Multiple CUDA devices/contexts, Linux, or other GPU architectures | **Not tested** |

[Machine-readable record](validation/host-checks.json) and [CTest log](validation/host-ctest.log) identify the checked host sources and actual outcomes.

The oracle host tests verify the CPU oracle against an analytic uniform-attention
answer and page permutation/causality properties. They do not execute the CUDA
kernels. The historical service used an RTX 4060 Laptop (`sm_89`); its results and
source lineage are recorded separately in [provenance](docs/provenance.md).

## Offline checks (no CUDA toolkit or GPU needed)

Requires CMake 3.20+, a C++17 compiler, and Python 3.9+ with only its standard library.
Run from the repository root:

```sh
cmake -S . -B build-host -DKVB_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-host --parallel 2
ctest --test-dir build-host --output-on-failure
python3 tools/validate_release.py
```

For an additional host memory/undefined-behavior check on Clang or GCC:

```sh
cmake -S . -B build-host-sanitize -DKVB_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-host-sanitize --parallel 2
ctest --test-dir build-host-sanitize --output-on-failure
```

These checks do not access the network, load model weights or contact another
machine. Generated build directories are not release contents.

## Next CUDA qualification

The first outstanding item is a clean NVCC build of the exact manifest revision,
followed by GPU self-test, memory checking, and only then bounded timing. The
commands in [reproducibility](docs/reproducibility.md) are a procedure to validate,
not a record of successful execution.

During release preparation, no SSH, Windows command, GPU/model run, remote
allocation, budget change or safety-ledger change was made. The originating
project's GPU controller does not accept this standalone executable without a
reviewed adapter. That integration is outside this source release; it must not be
bypassed to turn a pending result into a claim.
