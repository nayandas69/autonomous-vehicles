"""Optional matplotlib views; importing the core package needs no plotting backend."""

import numpy as np


def plot_scene(scenario, state=None, actors=None, ax=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    state = scenario.initial if state is None else state
    actors = scenario.actors_at(state.time) if actors is None else actors
    for offset, style in (
        (-scenario.road.width / 2, "k-"),
        (scenario.road.width / 2, "k-"),
        (0, "k--"),
    ):
        poses = scenario.road.samples(spacing=1, lateral=offset)
        ax.plot([p.x for p in poses], [p.y for p in poses], style, linewidth=1)
    ego = state.footprint(scenario.parameters)
    ax.add_patch(Polygon([[p.x, p.y] for p in ego.corners], color="tab:blue", alpha=0.8))
    for actor in actors:
        ax.add_patch(
            Polygon([[p.x, p.y] for p in actor.footprint.corners], color="tab:red", alpha=0.7)
        )
        ax.text(actor.pose.x, actor.pose.y + actor.width, actor.identifier, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title(f"{scenario.name}, t={state.time:.1f} s")
    ax.autoscale_view()
    return ax


def plot_episode(episode, scenario):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    plot_scene(scenario, ax=axes[0, 0])
    frames = episode.frames
    times = [frame.state.time for frame in frames]
    axes[0, 0].plot(
        [frame.state.pose.x for frame in frames],
        [frame.state.pose.y for frame in frames],
        label="Rear axle path",
    )
    axes[0, 0].legend()
    axes[0, 1].plot(times, [frame.state.speed for frame in frames], label="Measured")
    axes[0, 1].plot(times, [frame.target_speed for frame in frames], "--", label="Target")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[0, 1].legend()
    axes[1, 0].plot(times, [frame.lateral_error for frame in frames])
    axes[1, 0].set_ylabel("Lateral error (m)")
    axes[1, 1].plot(times, [frame.state.acceleration for frame in frames])
    axes[1, 1].set_ylabel("Acceleration (m/s²)")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"{episode.scenario_name}: {episode.termination}")
    fig.tight_layout()
    return fig


def plot_segmentation(image, target, prediction, class_names=("background", "road", "obstacle")):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    palette = ListedColormap(["#87b6a7", "#565966", "#e0524d"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    axes[0].imshow(np.moveaxis(image, 0, -1))
    for ax, mask in zip(axes[1:], (target, prediction), strict=True):
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        ax.imshow(mask, cmap=palette, vmin=0, vmax=len(class_names) - 1)
    for ax, title in zip(axes, ("RGB", "Ground truth", "Prediction"), strict=True):
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_history(history):
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("History cannot be empty")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set_ylabel("Training loss")
    axes[1].plot(epochs, [row["mean_iou"] for row in history], marker="o")
    axes[1].set_ylabel("Validation mean IoU")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_candidates(candidates, road):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    reference = road.samples()
    ax.plot([p.x for p in reference], [p.y for p in reference], "k--", label="Reference")
    for candidate in candidates:
        positions = candidate.path.positions()
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color="tab:green" if candidate.feasible else "tab:red",
            alpha=0.7,
            label=f"d={candidate.offset:g}: {candidate.reason}",
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8)
    return fig


def plot_calibration(report):
    import matplotlib.pyplot as plt

    bins = [item for item in report.bins if item.count]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", label="Calibrated")
    ax.plot(
        [item.confidence for item in bins],
        [item.accuracy for item in bins],
        marker="o",
        label=f"ECE={report.expected_error:.3f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Pixel accuracy")
    ax.legend()
    return fig


def plot_scan(scan, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    points = scan.points()
    ax.scatter(points[scan.hits, 0], points[scan.hits, 1], s=8, label="Returns")
    ax.scatter([scan.pose.x], [scan.pose.y], marker="^", label="Sensor")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    return ax
