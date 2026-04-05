from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class LabelEncoder:
    classes_: List[str]

    @property
    def class_to_id(self) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(self.classes_)}

    def transform(self, labels: Iterable[str]) -> List[int]:
        mapping = self.class_to_id
        transformed = []
        for label in labels:
            label_str = str(label)
            if label_str not in mapping:
                raise ValueError(f"Unknown label encountered during transform: {label_str}")
            transformed.append(mapping[label_str])
        return transformed

    def inverse_transform(self, indices: Iterable[int]) -> List[str]:
        return [self.classes_[int(idx)] for idx in indices]

    def to_dict(self) -> Dict[str, int]:
        return self.class_to_id


def build_label_encoder(labels: Iterable[str]) -> LabelEncoder:
    classes = sorted({str(label) for label in labels})
    return LabelEncoder(classes_=classes)


def encode_labels(labels: Iterable[str], encoder: LabelEncoder) -> List[int]:
    return encoder.transform(labels)
