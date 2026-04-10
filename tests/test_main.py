import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch

from buoy.main import _resolve_device, main

# --- _resolve_device ---


def test_resolve_device_cpu_returns_cpu():
    result = _resolve_device("cpu")
    assert result == "cpu"


def test_resolve_device_cpu_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _resolve_device("cpu")
    assert any("cpu" in str(w.message).lower() for w in caught)


def test_resolve_device_cuda_unavailable_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no GPU is available"):
        _resolve_device("cuda")


def test_resolve_device_none_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = _resolve_device(None)
    assert result == "cpu"


def test_resolve_device_none_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = _resolve_device(None)
    assert result == "cuda"


# --- main ---


def _make_mock_model(sample_rate=2048.0, psd_length=32.0):
    model = MagicMock()
    model.sample_rate = sample_rate
    model.psd_length = psd_length
    return model


@patch("buoy.main.Amplfi")
@patch("buoy.main.Aframe")
def test_main_raises_on_sample_rate_mismatch(MockAframe, MockAmplfi, tmp_path):
    MockAframe.return_value = _make_mock_model(sample_rate=2048.0)
    MockAmplfi.side_effect = [
        _make_mock_model(sample_rate=4096.0),  # mismatch
        _make_mock_model(sample_rate=2048.0),
    ]

    with pytest.raises(ValueError, match="Sample rate mismatch"):
        main(events="GW150914", outdir=tmp_path, device="cpu")


@patch("buoy.main.get_data")
@patch("buoy.main.Amplfi")
@patch("buoy.main.Aframe")
def test_main_skips_event_when_outputs_exist(
    MockAframe, MockAmplfi, mock_get_data, tmp_path
):
    """With force=False and existing outputs, get_data should not be called."""
    model = _make_mock_model()
    MockAframe.return_value = model
    MockAmplfi.return_value = model

    event = "GW150914"
    datadir = tmp_path / event / "data"
    datadir.mkdir(parents=True)
    (datadir / "aframe_outputs.hdf5").touch()
    (datadir / "posterior_samples.dat").touch()

    main(events=event, outdir=tmp_path, device="cpu", force=False)

    mock_get_data.assert_not_called()


@patch("buoy.main.get_data")
@patch("buoy.main.Amplfi")
@patch("buoy.main.Aframe")
def test_main_skips_event_when_run_amplfi_false(
    MockAframe, MockAmplfi, mock_get_data, tmp_path
):
    """With run_amplfi=False, only aframe output needs to exist to skip."""
    model = _make_mock_model()
    MockAframe.return_value = model
    MockAmplfi.return_value = model

    event = "GW150914"
    datadir = tmp_path / event / "data"
    datadir.mkdir(parents=True)
    (datadir / "aframe_outputs.hdf5").touch()
    # posterior_samples.dat intentionally absent

    main(
        events=event,
        outdir=tmp_path,
        device="cpu",
        force=False,
        run_amplfi=False,
    )

    mock_get_data.assert_not_called()


@patch("buoy.main.get_data")
@patch("buoy.main.Amplfi")
@patch("buoy.main.Aframe")
def test_main_reprocesses_event_when_force_true(
    MockAframe, MockAmplfi, mock_get_data, tmp_path
):
    """With force=True, get_data is called even when outputs already exist."""
    import h5py
    import numpy as np

    model = _make_mock_model()
    MockAframe.return_value = model
    MockAmplfi.return_value = model
    mock_get_data.return_value = (
        np.zeros((1, 2, 4096)),
        ["H1", "L1"],
        0.0,
        1.0,
    )

    event = "GW150914"
    datadir = tmp_path / event / "data"
    datadir.mkdir(parents=True)
    with h5py.File(datadir / "aframe_outputs.hdf5", "w") as f:
        f.create_dataset("times", data=np.array([0.0, 1.0]))
        f.create_dataset("ys", data=np.zeros(2))
        f.create_dataset("timing_integrated", data=np.zeros(2))
        f.create_dataset("signif_integrated", data=np.zeros(2))
    (datadir / "posterior_samples.dat").touch()

    main(
        events=event,
        outdir=tmp_path,
        device="cpu",
        force=True,
        run_aframe=False,
        run_amplfi=False,
        generate_plots=False,
    )

    mock_get_data.assert_called_once()
