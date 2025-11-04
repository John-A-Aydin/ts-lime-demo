import numpy as np

from tsmule.sampling.segment import AbstractSegmentation  # type: ignore  # noqa: E402


class UniformSegmentation(AbstractSegmentation):
    def __init__(self, partitions):
        if partitions < 2:
            raise ValueError("Number of partitions must be at least 2")
        self.partitions = partitions

    def segment(self, time_series_sample, **kwargs):
        n_steps = len(time_series_sample)
        mask = []

        segment_size = n_steps // self.partitions
        remainder = n_steps % self.partitions

        current_idx = 0
        for i in range(self.partitions):
            # Distributing the remainder accross first few segments
            current_segment_size = segment_size + (1 if i < remainder else 0)
            mask.extend([[i]] * current_segment_size)
            current_idx += current_segment_size

        return np.array(mask)
