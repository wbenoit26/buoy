import pytest
import torch
from conftest import FDURATION, NUM_CHANNELS, SAMPLE_RATE

from buoy.utils.data import slice_amplfi_data

KERNEL_LENGTH = 4.0
PSD_LENGTH = 32.0

EVENT_POSITION = KERNEL_LENGTH / 2  # event at centre of kernel


def make_data_and_times(event_position=EVENT_POSITION):
    """
    Build a strain tensor and t0/tc values sized to fit exactly one
    valid AMPLFI window with the given event_position.
    """
    total_seconds = PSD_LENGTH + KERNEL_LENGTH + FDURATION
    total_samples = int(total_seconds * SAMPLE_RATE)
    t0 = 0.0
    # choose tc so window_start lands exactly at the psd boundary
    tc = t0 + PSD_LENGTH + event_position + FDURATION / 2
    data = torch.arange(total_samples, dtype=torch.float32).expand(
        1, NUM_CHANNELS, -1
    )
    return data, t0, tc


def test_slice_amplfi_data_output_shapes():
    data, t0, tc = make_data_and_times()
    psd_data, window = slice_amplfi_data(
        data=data,
        sample_rate=SAMPLE_RATE,
        t0=t0,
        tc=tc,
        kernel_length=KERNEL_LENGTH,
        event_position=EVENT_POSITION,
        psd_length=PSD_LENGTH,
        fduration=FDURATION,
    )
    assert psd_data.shape == (NUM_CHANNELS, int(PSD_LENGTH * SAMPLE_RATE))
    assert window.shape == (
        NUM_CHANNELS,
        int((KERNEL_LENGTH + FDURATION) * SAMPLE_RATE),
    )


def test_slice_amplfi_data_psd_and_window_are_contiguous():
    """psd_data should end exactly where window begins."""
    data, t0, tc = make_data_and_times()
    psd_data, window = slice_amplfi_data(
        data=data,
        sample_rate=SAMPLE_RATE,
        t0=t0,
        tc=tc,
        kernel_length=KERNEL_LENGTH,
        event_position=EVENT_POSITION,
        psd_length=PSD_LENGTH,
        fduration=FDURATION,
    )
    # data is arange, so psd_data[-1] + 1 == window[0]
    assert torch.allclose(psd_data[:, -1] + 1, window[:, 0])


def test_slice_amplfi_data_window_out_of_bounds():
    total_samples = int(10 * SAMPLE_RATE)
    data = torch.zeros(1, NUM_CHANNELS, total_samples)
    with pytest.raises(ValueError):
        slice_amplfi_data(
            data=data,
            sample_rate=SAMPLE_RATE,
            t0=0.0,
            tc=0.1,
            kernel_length=KERNEL_LENGTH,
            event_position=EVENT_POSITION,
            psd_length=PSD_LENGTH,
            fduration=FDURATION,
        )
