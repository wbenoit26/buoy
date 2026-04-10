import pytest
import torch
from conftest import FDURATION, FFTLENGTH, HIGHPASS, NUM_CHANNELS, SAMPLE_RATE

from buoy.utils.preprocessing import (
    BackgroundSnapshotter,
    BatchWhitener,
    PsdEstimator,
)

INFERENCE_SAMPLING_RATE = 32
KERNEL_LENGTH = 1.0
PSD_LENGTH = 8.0
BATCH_SIZE = 8


def make_batch_whitener(**kwargs):
    defaults = {
        "kernel_length": KERNEL_LENGTH,
        "sample_rate": SAMPLE_RATE,
        "inference_sampling_rate": INFERENCE_SAMPLING_RATE,
        "batch_size": BATCH_SIZE,
        "fduration": FDURATION,
        "fftlength": FFTLENGTH,
        "highpass": HIGHPASS,
    }
    return BatchWhitener(**{**defaults, **kwargs})


def whitener_input(whitener):
    """Minimum-sized input for a BatchWhitener forward pass."""
    fsize = int(FDURATION * SAMPLE_RATE)
    kernel = (
        (BATCH_SIZE - 1) * whitener.stride_size + whitener.kernel_size + fsize
    )
    min_background = int(FFTLENGTH * SAMPLE_RATE)
    return torch.randn(NUM_CHANNELS, kernel + min_background)


# --- BackgroundSnapshotter ---


def test_snapshotter_output_shapes():
    snapshotter = BackgroundSnapshotter(
        psd_length=PSD_LENGTH,
        kernel_length=KERNEL_LENGTH,
        fduration=FDURATION,
        sample_rate=SAMPLE_RATE,
        inference_sampling_rate=INFERENCE_SAMPLING_RATE,
    )
    stride = int(SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
    update = torch.randn(1, NUM_CHANNELS, stride)
    snapshot = torch.zeros(1, NUM_CHANNELS, snapshotter.state_size)

    x, new_snapshot = snapshotter(update, snapshot)

    assert x.shape == (1, NUM_CHANNELS, snapshotter.state_size + stride)
    assert new_snapshot.shape == snapshot.shape


def test_snapshotter_state_is_tail_of_output():
    snapshotter = BackgroundSnapshotter(
        psd_length=PSD_LENGTH,
        kernel_length=KERNEL_LENGTH,
        fduration=FDURATION,
        sample_rate=SAMPLE_RATE,
        inference_sampling_rate=INFERENCE_SAMPLING_RATE,
    )
    stride = int(SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
    update = torch.randn(1, NUM_CHANNELS, stride)
    snapshot = torch.randn(1, NUM_CHANNELS, snapshotter.state_size)

    x, new_snapshot = snapshotter(update, snapshot)

    assert torch.allclose(new_snapshot, x[:, :, -snapshotter.state_size :])


# --- PsdEstimator ---


def test_psd_estimator_output_shapes():
    estimator = PsdEstimator(
        length=KERNEL_LENGTH,
        sample_rate=SAMPLE_RATE,
        fftlength=FFTLENGTH,
    )
    min_background = int(FFTLENGTH * SAMPLE_RATE)
    x = torch.randn(NUM_CHANNELS, estimator.size + min_background)

    data, psd = estimator(x)

    assert data.shape[-1] == estimator.size
    assert psd.shape[0] == NUM_CHANNELS


def test_psd_estimator_returns_tail_as_data():
    estimator = PsdEstimator(
        length=KERNEL_LENGTH,
        sample_rate=SAMPLE_RATE,
        fftlength=FFTLENGTH,
    )
    min_background = int(FFTLENGTH * SAMPLE_RATE)
    x = torch.randn(NUM_CHANNELS, estimator.size + min_background)

    data, _ = estimator(x)

    assert torch.allclose(data, x[..., -estimator.size :])


# --- BatchWhitener ---


def test_batch_whitener_returns_tensor_with_correct_shape():
    whitener = make_batch_whitener()
    x = whitener_input(whitener)

    result = whitener(x)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (BATCH_SIZE, NUM_CHANNELS, whitener.kernel_size)


def test_batch_whitener_return_whitened_returns_tuple():
    whitener = make_batch_whitener(return_whitened=True)
    x = whitener_input(whitener)

    batches, whitened = whitener(x)

    assert isinstance(batches, torch.Tensor)
    assert isinstance(whitened, torch.Tensor)
    assert batches.shape == (BATCH_SIZE, NUM_CHANNELS, whitener.kernel_size)


def test_batch_whitener_raises_on_wrong_ndim():
    whitener = make_batch_whitener()
    with pytest.raises(ValueError, match="2 or 3 dimensional"):
        whitener(torch.zeros(4))
    with pytest.raises(ValueError, match="2 or 3 dimensional"):
        whitener(torch.zeros(1, 2, 3, 4))
