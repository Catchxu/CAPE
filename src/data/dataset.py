from typing import Dict

from torch.utils.data import Dataset


class DictionaryTensorDataset(Dataset):
    def __init__(self, tensors: Dict[str, object]):
        self.tensors = tensors
        first_key = next(iter(tensors))
        self.length = int(tensors[first_key].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return {name: tensor[index] for name, tensor in self.tensors.items()}
