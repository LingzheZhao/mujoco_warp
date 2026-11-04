# NablaMimic analytic-adjoint integration

This branch preserves the SDK package used by the completed NablaMimic COM and
checkpoint evaluations. It changes dependency delivery: the five previously
applied diffs are now independent commits. The package name, API and upstream
version string remain `mujoco-warp`, `mujoco_warp` and `3.12.0`; that version string
alone does not identify this integration. Consumers must pin a complete Git
revision, or an explicitly versioned build of that revision.

The fork's `main` tracks official upstream code. `codex/adjoint` is the review base
for the adjoint integration; `codex/nablamimic-adjoint` contains the imported
changes. Released commits and tags must remain available for prior experiments.
Upstream synchronization is a separate reviewed change, not part of this migration.

## Sources and behavior

The unchanged base is the `adjoint` implementation at `357a75d`, from
[google-deepmind/mujoco_warp#1535](https://github.com/google-deepmind/mujoco_warp/pull/1535),
followed by the two commits in
[etaoxing/mujoco_warp#1](https://github.com/etaoxing/mujoco_warp/pull/1): pyramidal-contact
skip handling and joint actuator-force clamp derivatives. Both PRs were open
when this integration was prepared; this branch does not represent an official
upstream differentiability release.

The following five changes are separate commits with their source URLs:

| Historical identifier | Source and purpose |
| --- | --- |
| 0001 | Official NablaMimic: contact-normal derivative choices and a clamped-actuator qvel derivative correction option |
| 0002 | Official NablaMimic: freeze capsule segment parameters in the backward pass near parallel contact axes |
| 0003 | Official NablaMimic: optional soft actuator-clamp surrogate gradients |
| 0004 | Official NablaMimic: stable adjoint linear solves through RHS scaling or an unbordered factorization |
| 0005 | COM-only backport of merged upstream [#1872](https://github.com/google-deepmind/mujoco_warp/pull/1872), with subtree metadata from [#1730](https://github.com/google-deepmind/mujoco_warp/pull/1730) |

The first four changes originate in
[official NablaMimic](https://github.com/OamIcs/NablaMimic/tree/bfbb2a6773b4776c17402a34a93ec7589d9c7c18/patches/mujoco_warp).
The fifth is the [validated COM backport](https://github.com/LingzheZhao/NablaMimic/pull/5).
Source comments retain their historical patch identifiers and modification notices.
No diff application is needed to use this branch.

| Environment option | Behavior when enabled |
| --- | --- |
| `MJW_CLAMP_KV_FIX=1` | Correct the qvel derivative for clamped actuator forces |
| `MJW_ADJ_NORMAL=k`, `MJW_ADJ_NORMAL_EPS` | Contact-normal derivative: 0 exact, 1 held constant, 2 denominator regularized, 3 held constant below the distance threshold (default 1e-3 m) |
| `MJW_ADJ_PARALLEL=1`, `MJW_ADJ_PARALLEL_EPS` | Hold segment parameters constant for near-parallel capsule axes (default threshold 1.2e-3) |
| `MJW_CLAMP_SOFT=delta` | Soft surrogate clamp derivative; disabled at zero |
| `MJW_IFT_SOLVE=k` | 1 scales the adjoint RHS by a power of two; 2 uses the unbordered factorization |

These derivative options are off by default and leave forward dynamics unchanged.
NablaMimic sets them before importing the SDK; its alternate Warp training recipe
uses normal=3, parallel=1 and IFT=1. Its current NablaEngine training path does not
execute this SDK. Official evaluation uses the default derivative options, while
its constructor still executes the old forward/backward warmup.

COM range sums always apply in this branch. They preserve the old `mass != 0`
division rule; zero mass keeps the weighted sum. The newer `xipos` fallback is
excluded. RNE, CRB, inertia/axis calculations and old analytic-adjoint interfaces
are unchanged. Whole-simulator determinism and complete training validation of
every Warp recipe are not claimed. See the
[NablaMimic results](https://github.com/LingzheZhao/NablaMimic/blob/main/COM_EVALUATION.md)
for the fixed-input and full-evaluation evidence and its limitations.

## Install and validate

Use NablaMimic's `scripts/setup_mujoco_warp.sh`, which pins this fork and its
complete revision. For a manual installation, check out the same declared
revision in a fresh directory, then install it without resolving new runtime
dependencies:

```bash
git clone https://github.com/LingzheZhao/mujoco_warp.git mujoco_warp_fork
cd mujoco_warp_fork
git checkout --detach <declared-commit>
python -m pip install --no-deps -e .
```

The measured environment uses MuJoCo 3.17.0, Warp 1.18.0 (CUDA 12 build), NumPy
2.5.4 and JAX 0.12.0.1. Keep the consumer's environment lock; installing the PyPI
package or the floating `main` branch does not select this adjoint integration.

The standalone forward-only COM check moved here without code changes:

```bash
python studies/com_position_check.py --out /results/fresh-com-check
```

Run it in a provisioned CUDA environment. It needs no NablaMimic policy, robot
assets or scheduler. Five cases cover branching, multiple roots, world mass,
zero mass and tiny positive mass. Each uses sixteen captured repetitions and
two FP64 references. This does not extend zero-mass adjoint support or establish
full evaluator determinism. Raw results stay at the selected fresh output path.
