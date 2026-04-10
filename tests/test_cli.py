from unittest.mock import MagicMock, patch

import jsonargparse
import pytest

from buoy.cli import cli
from buoy.main import main


def _make_parser():
    """Build the same parser the CLI uses, without invoking main."""
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main, fail_untyped=False, sub_configs=True)
    parser.add_argument("--config", action="config")
    return parser


# --- argument parsing ---


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        cli(["--help"])
    assert exc.value.code == 0


def test_cli_missing_required_args_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        cli([])
    assert exc.value.code != 0


def test_cli_parses_events_and_outdir(tmp_path):
    parser = _make_parser()
    args = parser.parse_args(
        ["--events", "GW150914", "--outdir", str(tmp_path)]
    )
    assert args.events == "GW150914"
    assert args.outdir == tmp_path


def test_cli_parses_optional_flags(tmp_path):
    parser = _make_parser()
    args = parser.parse_args(
        [
            "--events",
            "GW150914",
            "--outdir",
            str(tmp_path),
            "--device",
            "cpu",
            "--run_amplfi",
            "false",
        ]
    )
    assert args.device == "cpu"
    assert args.run_amplfi is False


# --- invocation ---


def _make_mock_model(sample_rate=2048.0, psd_length=32.0):
    model = MagicMock()
    model.sample_rate = sample_rate
    model.psd_length = psd_length
    return model


@patch("buoy.main.Amplfi")
@patch("buoy.main.Aframe")
def test_cli_invokes_main_with_parsed_args(MockAframe, MockAmplfi, tmp_path):
    """
    End-to-end: CLI parses args and calls main; skipped event confirms flow.
    """
    model = _make_mock_model()
    MockAframe.return_value = model
    MockAmplfi.return_value = model

    event = "GW150914"
    datadir = tmp_path / event / "data"
    datadir.mkdir(parents=True)
    (datadir / "aframe_outputs.hdf5").touch()
    (datadir / "posterior_samples.dat").touch()

    cli(["--events", event, "--outdir", str(tmp_path), "--device", "cpu"])

    MockAframe.assert_called_once()
    MockAmplfi.assert_called()
