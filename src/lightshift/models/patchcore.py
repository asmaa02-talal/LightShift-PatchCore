"""PatchCore model factory using Anomalib."""

from anomalib.models import Patchcore


def build_patchcore(
    backbone: str = "wide_resnet50_2",
    layers: tuple[str, ...] = ("layer2", "layer3"),
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
) -> Patchcore:
    """Build the PatchCore baseline model."""
    return Patchcore(
        backbone=backbone,
        layers=layers,
        pre_trained=True,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        pre_processor=True,
        post_processor=True,
        evaluator=True,
        visualizer=False,
    )
