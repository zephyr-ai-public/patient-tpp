from typing import Callable, Optional

import torch
from torch import Tensor
from easy_tpp.model.torch_model.torch_thinning import EventSampler


class ZaiEventSampler(EventSampler):
    """Event Sequence Sampler based on thinning algorithm, which corresponds to Algorithm 2 of
    The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process,
    https://arxiv.org/abs/1612.09328.

    The implementation uses code from https://github.com/yangalan123/anhp-andtt/blob/master/anhp/esm/thinning.py.
    """

    def __init__(
        self,
        num_sample: int,
        num_exp: int,
        over_sample_rate: float,
        num_samples_boundary: int,
        dtime_max: float,
        patience_counter: int,
        device: int,
    ) -> None:
        """Initialize the event sampler.

        Args:
            num_sample (int): number of sampled next event times via thinning algo for computing predictions.
            num_exp (int): number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
            over_sample_rate (float): multiplier for the intensity up bound.
            num_samples_boundary (int): number of sampled event times to compute the boundary of the intensity.
            dtime_max (float): max value of delta times in sampling
            patience_counter (int): the maximum iteration used in adaptive thinning.
            device (torch.device): torch device index to select.
        """
        super(EventSampler, self).__init__()
        self.num_sample = num_sample
        self.num_exp = num_exp
        self.over_sample_rate = over_sample_rate
        self.num_samples_boundary = num_samples_boundary
        self.dtime_max = dtime_max
        self.patience_counter = patience_counter
        self.device = device

    def compute_intensity_upper_bound(
        self,
        time_seq: Tensor,
        time_delta_seq: Tensor,
        event_seq: Tensor,
        invars: Tensor,
        intensity_fn: Callable,
        compute_last_step_only: bool,
        numeric_layer: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the upper bound of intensity at each event timestamp.

        Args:
            time_seq (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            invars (tensor): [batch_size, ~3] time-invariant features.
            intensity_fn (fn): a function that computes the intensity.
            compute_last_step_only (bool): whether to compute the last time step only.
            numeric_layer (tensor): [batch_size, seq_len, hidden_size] Supplemental quantitative feature layer computed
                on its own forward pass

        Returns:
            tensor: [batch_size, seq_len]
        """
        batch_size, seq_len = time_seq.size()

        # [1, 1, num_samples_boundary]
        time_for_bound_sampled = torch.linspace(
            start=0.0, end=1.0, steps=self.num_samples_boundary, device=self.device
        )[None, None, :]

        # [batch_size, seq_len, num_samples_boundary]
        dtime_for_bound_sampled = time_delta_seq[:, :, None] * time_for_bound_sampled

        # [batch_size, seq_len, num_samples_boundary, event_num]
        intensities_for_bound = intensity_fn(
            time_seq,
            time_delta_seq,
            event_seq,
            invars,
            dtime_for_bound_sampled,
            max_steps=seq_len,
            compute_last_step_only=compute_last_step_only,
            numeric_layer=numeric_layer,
        )

        # [batch_size, seq_len]
        bounds = (
            intensities_for_bound.sum(dim=-1).max(dim=-1)[0] * self.over_sample_rate
        )

        return bounds

    def draw_next_time_one_step(
        self,
        time_seq: Tensor,
        time_delta_seq: Tensor,
        event_seq: Tensor,
        invars: Tensor,
        _dtime_boundary: Tensor,
        intensity_fn: Callable,
        compute_last_step_only: bool = False,
        numeric_layer: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute next event time based on Thinning algorithm.

        Args:
            time_seq (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            invars (tensor): [batch_size, ~3] time-invariant features.
            dtime_boundary (tensor): [batch_size, seq_len], dtime upper bound.
            intensity_fn (fn): a function to compute the intensity.
            compute_last_step_only (bool, optional): whether to compute last event timestep only. Defaults to False.
            numeric_layer (tensor): [batch_size, seq_len, hidden_size] Supplemental quantitative feature layer computed
                on its own forward pass

        Returns:
            tuple: next event time prediction and weight.
        """
        # 1. compute the upper bound of the intensity at each timestamp
        # the last event has no label (no next event), so we drop it
        # [batch_size, seq_len=max_len - 1]
        intensity_upper_bound = self.compute_intensity_upper_bound(
            time_seq,
            time_delta_seq,
            event_seq,
            invars,
            intensity_fn,
            compute_last_step_only,
        )

        # 2. draw exp distribution with intensity = intensity_upper_bound
        # we apply fast approximation, i.e., re-use exp sample times for computation
        # [batch_size, seq_len, num_exp]
        exp_numbers = self.sample_exp_distribution(intensity_upper_bound)
        exp_numbers = torch.cumsum(exp_numbers, dim=-1)

        # 3. compute intensity at sampled times from exp distribution
        # [batch_size, seq_len, num_exp, event_num]
        intensities_at_sampled_times = intensity_fn(
            time_seq,
            time_delta_seq,
            event_seq,
            invars,
            exp_numbers,
            max_steps=time_seq.size(1),
            compute_last_step_only=compute_last_step_only,
            numeric_layer=numeric_layer,
        )

        # [batch_size, seq_len, num_exp]
        total_intensities = intensities_at_sampled_times.sum(dim=-1)

        # add one dim of num_sample: re-use the intensity for samples for prediction
        # [batch_size, seq_len, num_sample, num_exp]
        total_intensities = torch.tile(
            total_intensities[:, :, None, :], [1, 1, self.num_sample, 1]
        )

        # [batch_size, seq_len, num_sample, num_exp]
        exp_numbers = torch.tile(exp_numbers[:, :, None, :], [1, 1, self.num_sample, 1])

        # 4. draw uniform distribution
        # [batch_size, seq_len, num_sample, num_exp]
        unif_numbers = self.sample_uniform_distribution(intensity_upper_bound)

        # 5. find out accepted intensities
        # [batch_size, seq_len, num_sample]
        res = self.sample_accept(
            unif_numbers, intensity_upper_bound, total_intensities, exp_numbers
        )

        # [batch_size, seq_len, num_sample]
        weights = torch.ones_like(res) / res.shape[2]

        # add a upper bound here in case it explodes, e.g., in ODE models
        return res.clamp(max=1e5), weights
