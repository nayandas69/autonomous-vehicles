# Data and numerical contracts

## Coordinates and units

| Domain | Contract |
| --- | --- |
| Planar world | meters, seconds, radians; x forward at yaw zero, y left |
| Vehicle pose | rear axle; body center offset derived from rear overhang |
| Scripted actor pose | body center |
| Camera frame | x right, y down, z forward; z-depth in meters |
| Image boxes | half-open continuous XYXY edges, positive area |
| RGB tensors | float32 CHW/BCHW in [0,1] for the synthetic pipeline |
| Segmentation masks | int64 HW/BHW class IDs 0=background, 1=road, 2=obstacle |
| Manifest image files | HWC uint8 RGB `.npy`, normalized by the loader |
| Manifest masks | integer HW `.npy`; labels must be within declared range |

## Simulation events

Rectangle touching counts as collision. Full body corners determine road departure.
Road extent is finite; polyline endpoint containment uses distance to the finite
centerline. This gives rounded endpoint regions rather than a rectangular end cap.
The driver uses perfect ground-truth actor states. Crossing actors may not be
considered until they enter its lateral corridor. This stopping rule is conservative
for stationary actors and is not a general trajectory predictor.

Episodes stop on collision, road departure, route completion, or duration. The
simulation is deterministic for the same configuration. A repeat of the same deterministic
scenario is not a new independent statistical sample. Discrete substeps do not guarantee
continuous collision detection for arbitrary fast motion.

## Metrics

Segmentation aggregates confusion counts before IoU. Classes with empty union have
null IoU and do not enter mean IoU. The streaming meter separately tracks ignored
labels and returns null summary values when no pixels were evaluated. It must be
merged only with meters sharing class count and ignore-label policy.

Detection uses score-ordered greedy class-aware matching. Tracking uses greedy IoU
association and a constant-velocity box prediction. Its MOTA-like summary is specific
to this implementation and has no benchmark compatibility claim. A switch is counted
when a previously matched truth ID later receives a different track ID.

Depth metrics mask invalid target depth. Invalid predicted depths on valid targets
are errors. Median scale alignment is an optional, explicitly separate experiment;
raw metric depth and aligned depth must not be mixed in comparisons.

Driving actor clearance is a circumscribed-disc lower bound and may be negative
without a rectangular collision. Lateral RMSE uses sampled frames; mean speed is
integrated over timestamps. Initial acceleration is zero by default, so jerk includes
the controller's initial transition.

## Data isolation

Generator seeds derive from base seed, split ID and index. Real data manifests require
unique sample IDs, unique image paths, and one split per scene. Relative paths cannot
escape the manifest directory, including via symlinks. Files use `allow_pickle=False`.
Manifest validation checks declared files and scene isolation, not semantic annotation
quality or content-level duplicate detection between differently named images.

The observation buffer returns only past samples and rejects non-increasing timestamps.
Maximum age determines freshness. A not-ready synchronization result contains null
values for unavailable sensors; callers must handle that state explicitly.

## Artifacts and reproducibility

Each run gets a fresh directory. Atomic JSON writes reject nonfinite values and
refuse overwrite unless explicitly requested by an API caller. A failed run can leave
partial artifacts for inspection. Checkpoints should be trusted local research artifacts;
restricted weights loading does not establish provenance or bound memory consumption.

The scoped RNG helper restores Python, NumPy, Torch and thread settings. It is
process-global and must not run concurrently from multiple threads. Bitwise
repeatability across platforms or dependency versions is not promised. GPU execution
is outside the tested CPU profile.
