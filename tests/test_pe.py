from unittest.mock import MagicMock

import torch

from buoy.utils.pe import filter_samples

INFERENCE_PARAMS = ["phi", "dec", "chirp_mass"]


class _BoundedPrior:
    """Prior that accepts [0, 1] and rejects everything else."""

    def log_prob(self, x):
        in_range = (x >= 0) & (x <= 1)
        return torch.where(
            in_range, torch.zeros_like(x), torch.full_like(x, float("-inf"))
        )


def _make_sampler(*param_names):
    sampler = MagicMock()
    sampler.priors = {p: _BoundedPrior() for p in param_names}
    return sampler


def test_filter_samples_removes_out_of_prior():
    """Samples outside any prior bound should be removed."""
    # 10 samples: first 5 in [0,1], last 5 outside
    samples = torch.cat(
        [
            torch.full((5, 3), 0.5),  # in range
            torch.full((5, 3), 2.0),  # out of range
        ]
    )
    sampler = _make_sampler(*INFERENCE_PARAMS)

    result = filter_samples(samples, sampler, INFERENCE_PARAMS)

    assert result.shape[0] == 5
    assert (result <= 1.0).all()


def test_filter_samples_all_in_prior():
    """When all samples are in-prior, none should be removed."""
    samples = torch.full((20, 3), 0.5)
    sampler = _make_sampler(*INFERENCE_PARAMS)

    result = filter_samples(samples, sampler, INFERENCE_PARAMS)

    assert result.shape[0] == 20


def test_filter_samples_mask_is_intersection():
    """A sample failing any single prior should be removed."""
    # column 0 (phi) in range, column 1 (dec) out of range
    samples = torch.tensor(
        [
            [0.5, 2.0, 0.5],  # fails dec
            [0.5, 0.5, 0.5],  # passes all
        ]
    )
    sampler = _make_sampler(*INFERENCE_PARAMS)

    result = filter_samples(samples, sampler, INFERENCE_PARAMS)

    assert result.shape[0] == 1
    assert torch.allclose(result, torch.tensor([[0.5, 0.5, 0.5]]))
