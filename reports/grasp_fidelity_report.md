# SO-101 grasp fidelity investigation

Date: November 7, 2026. Baseline: `df07805`, including grip improvement `880608d`.

The best next step is hardware calibration of contact geometry, clamp force, and material pairs. Increasing simulated grip strength further is not yet justified. Current grasps already survive the tested loads; the unresolved question is whether their forces, slip, failure boundaries, and release behavior match the user's arm.

No production physics defaults changed in this investigation. Three parallel investigations covered primary-source research, model/geometry auditing, and grip experiments. Reproducible scripts and raw measurements accompany this report.

## Measured numerical sensitivity

Run from the repository root with the development dependencies installed:

```bash
uv run python scripts/grasp_fidelity_experiment.py --hold-s 30 --output reports/grasp_fidelity_results.json
```

This script reuses the existing supported-pick fixture and movement helpers. Each case starts with a fresh environment and seed 3. Cubes are supported during 100 control frames of closure, then lifted 10 cm, held for 30 simulated seconds, moved through x/y reversals, and released by opening followed by an 8 cm downward retreat. The 20 ms control period remains fixed when halving the physics timestep. Profiles are applied before reset. Finger IDs are captured before contact-dimension ablations, preserving the original grasp masks.

Four cases: 19.05 mm cube at 10 g; 24 mm cube at 100 g; 40 mm cube at 100 g; and 24 mm cube rotated 45 degrees at 100 g. Placement is the regression fixture's favorable 8 mm proximal pickup. These are four scenarios, not four independent random trials.

The following values come from `scripts/grasp_fidelity_experiment.py`, with raw results in `grasp_fidelity_results.json`. Maximum TCP-relative translation includes lift transients, settling, rotation-induced center movement, and reversals. It is not a pure friction-creep measurement.

| Profile | 19.05 mm / 10 g | 24 mm / 100 g | 40 mm / 100 g | Rotated 24 mm / 100 g |
|---|---:|---:|---:|---:|
| Baseline | 1.614 mm | 2.010 mm | 2.677 mm | 1.957 mm |
| `impratio=1000` | 1.191 mm | 1.974 mm | 2.001 mm | 1.950 mm |
| NoSlip, 3 iterations | 1.931 mm | 1.976 mm | 1.268 mm | 2.530 mm |
| Timestep 1.25 ms | 1.349 mm | 1.185 mm | 2.065 mm | 1.173 mm |
| 100 iterations, tolerance `1e-10` | 1.787 mm | 2.030 mm | 2.539 mm | 1.962 mm |
| Initial impedance `0.99` | 1.167 mm | 1.968 mm | 2.387 mm | 3.105 mm |
| `condim=3` | 1.776 mm | 2.007 mm | 2.861 mm | 1.970 mm |
| `condim=4` | 1.438 mm | 1.992 mm | 2.649 mm | 1.978 mm |
| Half gripper torque limit | 2.120 mm | 1.984 mm | 3.008 mm | 2.643 mm |
| Finger sliding friction `0.3` | 0.937 mm | 1.971 mm | 2.902 mm | 1.989 mm |
| Object sliding friction `0.1` | 1.614 mm | 2.010 mm | 2.677 mm | 1.957 mm |

All 44 cases acquired, had zero lost-grasp samples through loaded movement, and released after opening plus retreat. No MuJoCo warnings were recorded. Baseline maximum penetration across these cases was 0.849 mm; half timestep reduced that maximum to 0.657 mm. Measurements sample control boundaries, so they do not bound every physics substep's penetration or force. Summed contact normal forces can include internal or redundant contact loading and must not be interpreted directly as hardware pinch force.

The half-timestep profile is the strongest numerical candidate for further validation: maximum translation fell 16-41% across all four scenarios. It takes twice as many physics steps per command. Timing is recorded for reproducibility, but this sequential desktop run is not a performance benchmark. A tighter solver did not consistently improve trajectories. Changing contact dimension had little effect in these cube tests, which do not challenge rotational resistance adequately.

The stationary-hold endpoint metric changes the ranking. The following displacement between the first and last hold samples comes from the same script and raw JSON:

| Profile | 19.05 mm / 10 g | 24 mm / 100 g | 40 mm / 100 g | Rotated 24 mm / 100 g |
|---|---:|---:|---:|---:|
| Baseline | 1.667 mm | 0.065 mm | 0.976 mm | 0.291 mm |
| `impratio=1000` | 0.580 mm | 0.052 mm | 0.162 mm | 0.160 mm |
| NoSlip, 3 iterations | 0.022 mm | 0.052 mm | 0.061 mm | 0.052 mm |
| Timestep 1.25 ms | 0.175 mm | 0.075 mm | 1.007 mm | 0.364 mm |

NoSlip is strongest for this hold endpoint metric, despite mixed maximum-translation results. Higher `impratio` improves hold drift in all four cases and remains a candidate for shared-backend validation. Half timestep reduces maximum relative translation in these four trajectories but does not consistently improve stationary drift. These endpoint distances include continued settling and do not bound excursions within the hold. Neither observable replaces contact-frame tangential velocity or object orientation traces.

## Physical fidelity gaps

### 1. Object material does not currently control finger friction

Both finger collision classes specify `priority=1`, sliding friction 1, torsional friction 0.005, and rolling friction 0.0005. Objects have lower priority. Consequently, finger settings select the contact friction. Changing object friction to 0.1 gave exactly the baseline trajectories and effective finger friction remained 1 in the experiment.

MuJoCo's [contact-parameter mixing rules](https://mujoco.readthedocs.io/en/latest/modeling.html#contact-parameters) explain this behavior. Merely removing priority is insufficient: equal-priority geoms take the maximum friction coefficient. Use measured material-pair coefficients, potentially explicit contact pairs generated for finger/object geom combinations, while retaining separately calibrated contact compliance. Avoid introducing a global material abstraction until the first calibrated pair and its runtime tests establish the needed contract.

The rotational coefficients have units of length. A 5 mm torsional coefficient and 0.5 mm rolling coefficient require justification from the actual pad/contact patch, rather than interpretation as dimensionless material coefficients. [MuJoCo contact dimensions](https://mujoco.readthedocs.io/en/latest/computation/index.html#condim).

### 2. Gripper force is assumed rather than identified

The vendored `so101.xml` uses `kp=998.22`, `kv=2.731`, and a gripper torque range of +/-2.94 Nm. The proportional term alone reaches that limit at about 0.169 degrees of position error. The test controller closes relative to measured position by about 11.46 degrees per control frame, making saturated actuator behavior especially relevant. Joint damping, friction loss, and armature also influence the actual contact force.

The XML comments trace these values to an [analytical motor-model derivation](https://raw.githubusercontent.com/Gregory119/RBE501-RL-arm-project/main/gymnasium_env/README.md), with an assumed servo gain. This is provenance, not hardware validation.

[LeRobot v0.7.0](https://raw.githubusercontent.com/huggingface/lerobot/v0.7.0/src/lerobot/robots/so_follower/so_follower.py), matching this project's dependency line, configures gripper `Max_Torque_Limit=500`, `Protection_Current=250`, and `Overload_Torque=25`. Those registers do not establish a linear conversion from the simulated torque limit. The half-torque experiment is a sensitivity check, not a calibrated LeRobot preset.

The [official hardware repository](https://github.com/TheRobotStudio/SO-ARM100#sourcing-parts) distinguishes servo/supply configurations. Its [optional hardware](https://github.com/TheRobotStudio/SO-ARM100#optional-hardware) includes compliant fingers and grip tape. Identify the actual motor, supply, firmware, protection settings, print material, and finger surface before claiming parity with a real SO-101.

### 3. Collision coverage improved at a measurable geometric cost

`scripts/grasp_geometry_audit.py` probes visible mesh triangles on a 0.5 mm grid against the compiled collider union. It extends the existing coverage test's measurement in both signed directions. Depth uses the minimum signed distance across individual colliders, a containment diagnostic rather than the exact distance to the union boundary where colliders overlap. Reproduce current and previous geometry with:

```bash
uv run python scripts/grasp_geometry_audit.py
uv run python scripts/grasp_geometry_audit.py --revision '880608d^'
```

Raw JSON is under `grasp_fidelity/`.

| Signed surface-distance diagnostic | Before `880608d` | Current |
|---|---:|---:|
| Fixed jaw worst uncovered gap | 1.663 mm | 0.137 mm |
| Moving jaw worst uncovered gap | 1.651 mm | 0.171 mm |
| Fixed jaw 95th percentile depth inside colliders | 0.220 mm | 1.085 mm |
| Moving jaw 95th percentile depth inside colliders | 0.191 mm | 1.056 mm |
| Fixed samples more than 0.3 mm inside colliders | 2.87% | 28.32% |
| Moving samples more than 0.3 mm inside colliders | 1.21% | 21.45% |

There are 1,116 fixed-jaw and 993 moving-jaw samples. These are signed distances at visible surface samples, not exact aperture errors or a full symmetric surface-distance measurement. The largest interior depths predate the recent change. Visual meshes are also not a substitute for physical measurements.

Keep the improved coverage, then investigate smaller convex pieces or measured pad primitives that reduce excess fill. Before changing geometry, add regression coverage for both missing and excessive collider volume, first-contact angle, opening versus jaw angle, and narrow-object acquisition. The current geometry test bounds uncovered surfaces only.

## Supplemental experiments and limitations

Supplemental scripts and raw data under `grasp_fidelity/` explore placement depth, friction, force limits, and stationary release. Their shorter protocol uses 100 closure, 50 lift, 200 hold, and 100 opening frames, without reversals. These 53 exploratory cases are separate from the main sweep. Reproduce them with:

```bash
uv run python reports/grasp_fidelity/geometry_probe.py
```

The command also regenerates stationary-release diagnostics in `release_debug.json`.

Across the initial five placement/load scenarios, half timestep reduced maximum translation from the baseline range of 0.983-1.108 mm to 0.650-0.823 mm. Raising initial impedance cut penetration but increased the rotated 100 g cube's translation from 0.983 to 2.068 mm. At an extreme distal placement all five numerical profiles failed acquisition with no finger contact. Because the supplementary placement implementation changes support dimensions after compilation without refreshing compiled bounding radii, treat distal failure boundaries as exploratory and recompile each scene before relying on them.

Negative controls responded to physical parameters: zero finger friction lost the 100/500 g cubes during lift, and 10% torque failed acquisition in tested cases. Favorable cube retention therefore does not establish a calibrated load limit.

At the default placement, some axis-aligned cubes stayed on the fixed finger after opening at a stationary TCP. The moving jaw reached approximately 100 degrees and no longer contacted the object; the fixed tip spheres supported it. This is consistent with a geometric perch, not evidence of adhesion. Compare physical tip shape and release direction before classifying it as a defect. The existing retreat-to-release test remains meaningful but does not guarantee immediate free fall after opening alone.

## Recommended path forward

1. **Define the hardware target and collect calibration data.** Measure opening and first-contact angle at several jaw depths using known blocks/cylinders. Record commanded/observed angles in degrees, supply voltage, current, firmware, motor variant, pad material, and protection settings.
2. **Identify free motion, then clamp force.** Fit command delay, velocity limits, gains, damping, and reversal backlash from unloaded steps/ramps. Measure clamp force versus opening, depth, position error, and hold duration with a force gauge. Include protection-related force decay. Do not fit unknown torque by increasing friction.
3. **Calibrate one material pair.** Measure pull-out force at multiple measured clamp forces, then rotational slip under known moments. Fit sliding and rotational resistance separately. Add runtime tests proving object material changes the effective contact and expected failure threshold.
4. **Refine collision geometry against measurements.** Preserve coverage while reducing excessive fill. Compare opening-versus-angle and first-contact location against the physical arm, including fingertip and off-center contacts.
5. **Establish numerical convergence.** Compare 2.5, 1.25, and 0.625 ms at the same control rate, evaluating force, object orientation, contact count, penetration, and steady slip. Test native/multiple-contact collision flags supported by the installed engine. Use half timestep as a candidate, not a new default based on four cases.
6. **Validate held-out failures as well as successes.** Use different shapes, masses, friction pairs, offsets, approach angles, and acceleration profiles. Include stationary release and retreat release. Repeat hardware trials and compare distributions of slip, rotation, pullout force, drop threshold, and release delay. Predefine tolerances from measurement repeatability and task needs.
7. **Repeat across backends.** CPU MuJoCo results do not establish Warp parity. Warp rejects NoSlip and tolerance below `1e-6` in the current integration. Validate supported settings and contact manifolds separately on Warp CPU and target GPU hardware.

MuJoCo's [preventing-slip guidance](https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip) separates insufficient friction/normal force, geometry, vibration, and residual soft-contact creep. Current defaults already use elliptic cones and high friction impedance. NoSlip is a residual-creep option with additional cost and possible instability, rather than a hardware model. [Direct stiffness/damping parameters](https://mujoco.readthedocs.io/en/latest/modeling.html#reference) can help identify compliance once measurements exist. Check installed-version capabilities before adopting features from the moving `latest` documentation.

Avoid increasing torque, friction, rolling resistance, or artificial attachment until measured hardware behavior requires it. The best faithful model must reproduce objects that the real gripper drops as well as those it holds.

## Validation

The existing loaded-grasp regression passed all 12 cases before the study. The 44-case main experiment completed on MuJoCo 3.17.0, with raw platform/version metadata in its JSON output. `make format lint typecheck` passed formatting and lint, then reported 24 type errors in unchanged library files (`envhub.py` and existing environment inheritance declarations). Scoped `ty check` passed for all three experiment scripts. The targeted geometry, grasp/targeting, and physics-config suite passed 86 tests, in addition to the 12 loaded-grasp tests. Core grasp and documentation consistency checks passed another 40 tests. Commands:

```bash
uv run pytest tests/mujoco/test_grasp_dynamics.py -q
uv run pytest tests/core/test_jaw_collision_geometry.py tests/mujoco/test_grasp_and_targeting.py tests/core/test_physics_config.py -q
uv run pytest tests/core/test_grasp.py tests/test_docs_consistency.py -q
make format lint typecheck
uv run ty check scripts/grasp_fidelity_experiment.py scripts/grasp_geometry_audit.py reports/grasp_fidelity/geometry_probe.py
```

The main experiment also completed a four-case very-short-hold check (`--hold-s 0.001 --profiles baseline`) and rejected `--hold-s 0`. A broader Warp/core/docs invocation was deliberately stopped after six Warp grasp cases passed in 125 seconds; the remaining Warp matrix was not completed. Core/docs checks were then run separately to completion as reported above. This is a Warp spot check, not validation of any new numerical profile on Warp.

No hardware experiments were available, so near-perfect real-world parity remains unverified.
