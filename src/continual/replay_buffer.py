"""
PART 4: Experience Replay Buffer
=================================
Maintains a fixed-size buffer of representative samples from
each previously learned domain. During training on a new domain,
30% of each batch is sampled from the replay buffer to
maintain competence on old domains.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import yaml

logger = logging.getLogger(__name__)


class DomainReplayBuffer:
    """
    Per-domain replay buffer using reservoir sampling.

    For each domain:
    - Stores up to 2,000 representative clips
    - Uses reservoir sampling for unbiased selection
    - Supports both tensor (in-memory) and disk storage

    During training on domain N:
    - 70% of batch = new domain data
    - 30% of batch = uniformly sampled from all previous buffers
    """

    def __init__(
        self,
        buffer_size_per_domain: int = 2000,
        replay_ratio: float = 0.3,
        sampling_strategy: str = "reservoir",
        storage_mode: str = "disk",
        clip_length: int = 8,
        clip_stride: int = 4,
        base_dir: str = "replay_buffer",
    ):
        self.buffer_size = buffer_size_per_domain
        self.replay_ratio = replay_ratio
        self.sampling_strategy = sampling_strategy
        self.storage_mode = storage_mode
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.base_dir = Path(base_dir)

        # In-memory buffers: domain_name -> list of samples
        self.buffers: Dict[str, List] = {}
        # Count of total samples seen (for reservoir sampling)
        self.sample_counts: Dict[str, int] = {}

        logger.info(
            f"Replay buffer: size={buffer_size_per_domain}/domain, "
            f"ratio={replay_ratio}, strategy={sampling_strategy}"
        )

    def add_sample(self, domain_name: str, sample):
        """
        Add a sample to the domain buffer using reservoir sampling.

        Reservoir sampling guarantees that after seeing N samples,
        each sample has exactly buffer_size/N probability of being in the buffer.
        This gives an unbiased representative subset without needing
        multiple passes over the data.

        Args:
            domain_name: Which domain this sample belongs to
            sample: The data sample (dict with images, labels, etc.)
        """
        if domain_name not in self.buffers:
            self.buffers[domain_name] = []
            self.sample_counts[domain_name] = 0

        self.sample_counts[domain_name] += 1
        n = self.sample_counts[domain_name]

        if len(self.buffers[domain_name]) < self.buffer_size:
            # Buffer not full yet — always add
            self.buffers[domain_name].append(sample)
        else:
            # Reservoir sampling: replace with probability buffer_size/n
            idx = random.randint(0, n - 1)
            if idx < self.buffer_size:
                self.buffers[domain_name][idx] = sample

    def populate_from_dataloader(self, domain_name: str, dataloader: DataLoader):
        """
        Fill the replay buffer from a full dataloader (typically after training).

        Args:
            domain_name: Domain identifier
            dataloader: DataLoader for the completed domain
        """
        logger.info(f"Populating replay buffer for domain: {domain_name}")

        for batch in dataloader:
            # Handle batched data — add each sample individually
            if isinstance(batch, dict):
                batch_size = next(iter(batch.values())).shape[0]
                for i in range(batch_size):
                    sample = {k: v[i].cpu() for k, v in batch.items()}
                    self.add_sample(domain_name, sample)
            elif isinstance(batch, (list, tuple)):
                batch_size = batch[0].shape[0]
                for i in range(batch_size):
                    sample = tuple(b[i].cpu() for b in batch)
                    self.add_sample(domain_name, sample)

        logger.info(
            f"Buffer for '{domain_name}': "
            f"{len(self.buffers[domain_name])}/{self.buffer_size} slots filled "
            f"(from {self.sample_counts[domain_name]} total samples)"
        )

    def get_replay_dataset(self, exclude_domain: Optional[str] = None) -> Optional[Dataset]:
        """
        Create a combined dataset from all replay buffers (except the current domain).

        Args:
            exclude_domain: Domain currently being trained (don't replay its own data)

        Returns:
            A PyTorch Dataset of replay samples, or None if no buffers exist
        """
        all_samples = []
        for domain_name, buffer in self.buffers.items():
            if domain_name == exclude_domain:
                continue
            all_samples.extend(buffer)

        if not all_samples:
            return None

        logger.info(
            f"Replay dataset: {len(all_samples)} samples from "
            f"{len(self.buffers) - (1 if exclude_domain in self.buffers else 0)} domains"
        )

        return ReplayDataset(all_samples)

    def get_mixed_dataloader(
        self,
        new_domain_dataset: Dataset,
        exclude_domain: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """
        Create a DataLoader that mixes new domain data with replay data.

        The replay_ratio controls the mixing:
        - replay_ratio=0.3 → 70% new, 30% old
        - This is done by concatenating and shuffling

        Args:
            new_domain_dataset: Dataset for the new domain
            exclude_domain: Domain being trained (exclude from replay)
            batch_size: Batch size
            num_workers: Number of data loading workers
            collate_fn: Custom batch collation function

        Returns:
            Mixed DataLoader
        """
        replay_dataset = self.get_replay_dataset(exclude_domain)

        if replay_dataset is None:
            # First domain — no replay data available
            return DataLoader(
                new_domain_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        # Calculate how many replay samples to include
        n_new = len(new_domain_dataset)
        n_replay = int(n_new * self.replay_ratio / (1 - self.replay_ratio))
        n_replay = min(n_replay, len(replay_dataset))

        # Subsample replay dataset if needed
        if n_replay < len(replay_dataset):
            indices = random.sample(range(len(replay_dataset)), n_replay)
            replay_subset = torch.utils.data.Subset(replay_dataset, indices)
        else:
            replay_subset = replay_dataset

        # Concatenate and create mixed loader
        mixed = ConcatDataset([new_domain_dataset, replay_subset])

        logger.info(
            f"Mixed dataloader: {n_new} new + {n_replay} replay = {len(mixed)} total"
        )

        return DataLoader(
            mixed,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def save(self, domain_name: str):
        """Save buffer for a domain to disk."""
        if domain_name not in self.buffers:
            return

        save_dir = self.base_dir / domain_name
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "buffer": self.buffers[domain_name],
                "sample_count": self.sample_counts[domain_name],
            },
            save_dir / "buffer.pt",
        )
        logger.info(f"Replay buffer saved: {save_dir}")

    def load(self, domain_name: str):
        """Load buffer for a domain from disk."""
        load_path = self.base_dir / domain_name / "buffer.pt"
        if not load_path.exists():
            logger.warning(f"No buffer found at {load_path}")
            return

        data = torch.load(load_path, map_location="cpu", weights_only=False)
        self.buffers[domain_name] = data["buffer"]
        self.sample_counts[domain_name] = data["sample_count"]
        logger.info(
            f"Replay buffer loaded: {domain_name} "
            f"({len(self.buffers[domain_name])} samples)"
        )

    def stats(self) -> Dict[str, dict]:
        """Return buffer statistics."""
        return {
            domain: {
                "buffer_size": len(buf),
                "max_size": self.buffer_size,
                "fill_ratio": len(buf) / self.buffer_size,
                "total_seen": self.sample_counts.get(domain, 0),
            }
            for domain, buf in self.buffers.items()
        }


class ReplayDataset(Dataset):
    """Simple Dataset wrapper for replay buffer samples."""

    def __init__(self, samples: List):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
