"""Declared seven-class partitions; no evaluation labels select a taxonomy.

The clinical partition tests the previous FSN paper's grouping. Random controls
are fixed *before* training and preserve its group-size profile, not semantics.
Canonical IDs come from the existing manifest parser rather than a new order.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, Mapping

from experiments.pilot_data import EXPECTED_LABELS


CLASS_NAMES = tuple(EXPECTED_LABELS[index] for index in range(len(EXPECTED_LABELS)))
NUM_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class Taxonomy:
    name: str
    groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        groups = tuple(tuple(int(label) for label in group) for group in self.groups)
        object.__setattr__(self, "groups", groups)
        labels = [label for group in groups for label in group]
        if not self.name or not groups or any(not group for group in groups):
            raise ValueError("taxonomy needs a name and nonempty groups")
        if sorted(labels) != list(range(NUM_CLASSES)):
            raise ValueError("taxonomy must partition each canonical label exactly once")

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(len(group) for group in self.groups)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def label_to_group(self) -> tuple[int, ...]:
        mapping = [0] * NUM_CLASSES
        for group_index, group in enumerate(self.groups):
            for label in group:
                mapping[label] = group_index
        return tuple(mapping)

    @property
    def label_to_conditional(self) -> tuple[int, ...]:
        mapping = [0] * NUM_CLASSES
        for group in self.groups:
            for within_group, label in enumerate(group):
                mapping[label] = within_group
        return tuple(mapping)

    @property
    def auxiliary_rows(self) -> int:
        """Group head plus learned fine heads; singleton fine heads do not exist."""
        return self.num_groups + sum(size for size in self.group_sizes if size > 1)

    def group_targets(self, labels):
        import torch

        _validate_targets(labels)
        mapping = torch.tensor(self.label_to_group, device=labels.device, dtype=torch.long)
        return mapping[labels]

    def conditional_targets(self, labels):
        import torch

        _validate_targets(labels)
        mapping = torch.tensor(self.label_to_conditional, device=labels.device, dtype=torch.long)
        return mapping[labels]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "groups": [list(group) for group in self.groups],
            "class_names": list(CLASS_NAMES),
            "group_sizes": list(self.group_sizes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Taxonomy":
        if "class_names" in data and tuple(data["class_names"]) != CLASS_NAMES:
            raise ValueError("taxonomy class names do not match canonical manifest IDs")
        return cls(str(data["name"]), tuple(tuple(group) for group in data["groups"]))


def _validate_targets(labels) -> None:
    import torch

    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("targets must be one-dimensional int64 canonical IDs")
    if labels.numel() and bool(((labels < 0) | (labels >= NUM_CLASSES)).any()):
        raise ValueError("target outside canonical seven-class mapping")


def clinical_taxonomy() -> Taxonomy:
    by_name = {name: index for index, name in enumerate(CLASS_NAMES)}
    groups = (("消毒",), ("固定",), ("进针", "运针", "拔针"), ("扫散", "再灌注"))
    return Taxonomy("clinical", tuple(tuple(by_name[name] for name in group) for group in groups))


def _partition_signature(groups: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(group)) for group in groups))


def matched_random_taxonomies(seeds: Iterable[int] = (17, 29, 43), *, preserve_singletons: bool = True) -> tuple[Taxonomy, ...]:
    """Fixed random controls, excluding the clinical partition and duplicates.

    The primary control preserves singleton labels (disinfection/fixation),
    avoiding a change in which labels bypass fine classification. Optional
    full permutations are a separate singleton-allocation sensitivity check.
    Rejection sampling only compares class-ID partitions; it never reads
    predictions, losses, source metadata, or train/validation/test labels.
    """
    clinical = clinical_taxonomy()
    seen = {_partition_signature(clinical.groups)}
    controls = []
    for seed in seeds:
        generator = random.Random(int(seed))
        for _ in range(10000):
            singleton_labels = {group[0] for group in clinical.groups if len(group) == 1}
            labels = [label for label in range(NUM_CLASSES)
                      if not preserve_singletons or label not in singleton_labels]
            generator.shuffle(labels)
            offset, groups = 0, []
            for original_group in clinical.groups:
                size = len(original_group)
                if preserve_singletons and size == 1:
                    groups.append(original_group)
                else:
                    groups.append(tuple(sorted(labels[offset:offset + size])))
                    offset += size
            signature = _partition_signature(groups)
            if signature not in seen:
                seen.add(signature)
                prefix = "random" if preserve_singletons else "random_all"
                controls.append(Taxonomy(f"{prefix}_{int(seed)}", tuple(groups)))
                break
        else:
            raise ValueError("cannot generate another distinct matched-size taxonomy")
    return tuple(controls)


def get_taxonomy(name: str = "clinical") -> Taxonomy:
    options = (clinical_taxonomy(),) + matched_random_taxonomies()
    for taxonomy in options:
        if taxonomy.name == name:
            return taxonomy
    raise ValueError(f"unknown declared taxonomy: {name}")
