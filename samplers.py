# samplers.py

import random
import torch

import random
from typing import Dict, Iterator, List, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Sampler

from abc import abstractmethod
from typing import List, Tuple

from torch import Tensor
from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder
import os

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path

class FewShotDataset(Dataset):
    """
    Abstract class for all datasets used in a context of Few-Shot Learning.
    The tools we use in few-shot learning, especially TaskSampler, expect an
    implementation of FewShotDataset.
    Compared to PyTorch's Dataset, FewShotDataset forces a method get_labels.
    This exposes the list of all items labels and therefore allows to sample
    items depending on their label.
    """

    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        raise NotImplementedError(
            "All PyTorch datasets, including few-shot datasets, need a __getitem__ method."
        )

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError(
            "All PyTorch datasets, including few-shot datasets, need a __len__ method."
        )

    @abstractmethod
    def get_labels(self) -> List[int]:
        raise NotImplementedError(
            "Implementations of FewShotDataset need a get_labels method."
        )

GENERIC_TYPING_ERROR_MESSAGE = (
    "Check out the output's type of your dataset's __getitem__() method."
    "It must be a Tuple[Tensor, int] or Tuple[Tensor, 0-dim Tensor]."
)


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it samples
    n_way classes, then n_shot + n_query samples per class.
    """

    def __init__(self, dataset, n_way, n_shot, n_query, n_tasks):
        super().__init__(data_source=None)
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label = {}
        for idx, label in enumerate(dataset.get_labels()):
            self.items_per_label.setdefault(label, []).append(idx)

        self._check_dataset_size_fits_sampler_parameters()

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            episode_indices = []
            selected_classes = random.sample(sorted(self.items_per_label.keys()), self.n_way)
            for label in selected_classes:
                indices = random.sample(self.items_per_label[label], self.n_shot + self.n_query)
                episode_indices.extend(indices)  # just indices, no loading data here
            yield episode_indices

    def episodic_collate_fn(self, input_data: List[Tuple[Tensor, int, str]]):
        true_class_ids = list({x[1] for x in input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape((self.n_way, self.n_shot + self.n_query, *all_images.shape[1:]))

        all_labels = torch.tensor([true_class_ids.index(x[1]) for x in input_data]).reshape((self.n_way, self.n_shot + self.n_query))

        all_paths = [x[2] for x in input_data]
        all_paths = [all_paths[i * (self.n_shot + self.n_query):(i + 1) * (self.n_shot + self.n_query)] for i in range(self.n_way)]

        support_images = all_images[:, :self.n_shot].reshape(-1, *all_images.shape[2:])
        support_labels = all_labels[:, :self.n_shot].flatten()
        support_paths = []
        for paths in all_paths:
            support_paths.extend(paths[:self.n_shot])

        query_images = all_images[:, self.n_shot:].reshape(-1, *all_images.shape[2:])
        query_labels = all_labels[:, self.n_shot:].flatten()
        query_paths = []
        for paths in all_paths:
            query_paths.extend(paths[self.n_shot:])

        # Convert full paths to just filenames
        support_paths = [os.path.basename(p) for p in support_paths]
        query_paths = [os.path.basename(p) for p in query_paths]

        # Sanity check lengths:
        assert len(query_paths) == query_images.size(0), f"Query paths length {len(query_paths)} != Query images {query_images.size(0)}"
        assert len(support_paths) == support_images.size(0), f"Support paths length {len(support_paths)} != Support images {support_images.size(0)}"

        return support_images, support_labels, query_images, query_labels, support_paths, query_paths

    @staticmethod
    def _cast_input_data_to_tensor_int_tuple(
        input_data: List[Tuple[Tensor, Union[Tensor, int], str]]
    ) -> List[Tuple[Tensor, int, str]]:
        result = []
        for image, label, path in input_data:
            if not isinstance(image, Tensor):
                raise TypeError(f"Invalid image type: {type(image)}")
            if isinstance(label, Tensor):
                if label.ndim != 0:
                    raise ValueError(f"Invalid label shape: {label.shape}")
                label = int(label.item())
            elif not isinstance(label, int):
                raise TypeError(f"Invalid label type: {type(label)}")
            result.append((image, label, path))
        return result

    def _check_dataset_size_fits_sampler_parameters(self):
        self._check_dataset_has_enough_labels()
        self._check_dataset_has_enough_items_per_label()

    def _check_dataset_has_enough_labels(self):
        if self.n_way > len(self.items_per_label):
            raise ValueError(
                f"Dataset has {len(self.items_per_label)} classes, but n_way={self.n_way}"
            )

    def _check_dataset_has_enough_items_per_label(self):
        for label, items in self.items_per_label.items():
            if len(items) < self.n_shot + self.n_query:
                raise ValueError(
                    f"Label {label} has {len(items)} items, but requires at least {self.n_shot + self.n_query}"
                )
