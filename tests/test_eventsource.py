"""Unit tests use decoded records only; no archive access is needed."""

from copy import deepcopy
from importlib.metadata import entry_points
import os

import astropy.units as u
import fzreader
import numpy as np
import pytest
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from ctapipe.containers import NAN_TIME, EventType, SubarrayEventContainer
from ctapipe.io import DataLevel, EventSource
from traitlets import TraitError
from traitlets.config import Config

from ctapipe_io_whipple import WhippleEventSource
from ctapipe_io_whipple.eventsource import create_camera_geometry, _event_time, _default_dc_to_pe


@pytest.fixture
def records():
    common = dict(record_was_decoded=True, run_num=5000, record_time_mjd=52800.5)
    event = dict(
        **common, record_type="event", event_num=0, event_type="sky",
        nadc=120, adc_values=list(range(120)), gps_is_good=True,
        gps_mjd=52800, gps_utc_sec=43200, gps_ns=123456789,
    )
    frame = dict(event, record_type="frame", frame_num=1, event_type="pedestal")
    frame.pop("event_num")
    return [
        dict(common, record_type="run", nominal_mjd_start=52800.5),
        dict(common, record_type="tracking", telescope_az_deg=180, telescope_el_deg=70),
        dict(record_type="ccd", record_was_decoded=False),
        event, frame, dict(event, event_num=1),
    ]


@pytest.fixture
def fake_file(tmp_path, monkeypatch, records):
    readers = []

    class Reader:
        def __init__(self, path):
            self.closed = False
            self.records = iter(deepcopy(records))
            readers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def __iter__(self):
            return self

        def __next__(self):
            assert not self.closed
            return next(self.records)

    monkeypatch.setattr(fzreader, "FZReader", Reader)
    path = tmp_path / "gt005000.fz"
    path.touch()
    return path, readers


@pytest.mark.parametrize("nadc,npix", [(120, 109), (156, 151), (336, 331), (492, 490), (384, 379)])
def test_camera(nadc, npix):
    geometry = create_camera_geometry(nadc)
    description = fzreader.get_camera_geometry_by_nadc(nadc)
    assert geometry.n_pixels == npix
    assert isinstance(geometry.frame, CameraFrame)

    geometry= geometry.transform_to(TelescopeFrame())
    np.testing.assert_allclose(geometry.pix_x.to_value(u.deg), description["x"])
    np.testing.assert_allclose(geometry.pix_y.to_value(u.deg), description["y"])
    np.testing.assert_allclose(geometry.pix_area.to_value(u.deg**2), np.pi * np.asarray(description["r"])**2)
    for pixel, neighbors in enumerate(description["neighbors"]):
        assert set(geometry.neighbors[pixel]) == set(neighbors)


def test_unknown_camera():
    with pytest.raises(ValueError, match="Unsupported"):
        create_camera_geometry(12)


def test_dummy_camera_readout():
    from ctapipe_io_whipple.eventsource import _create_subarray

    readout = _create_subarray(120).tel[1].camera.readout
    assert readout.sampling_rate == 1 * u.GHz
    assert readout.n_samples == 1
    assert readout.reference_pulse_shape.shape == (1, 1)


def test_events(fake_file):
    path, readers = fake_file
    with WhippleEventSource(path) as source:
        assert source.datalevels == (DataLevel.R0, DataLevel.R1)
        assert not source.is_simulation
        assert source.obs_ids == [5000]
        assert source.observation_blocks[5000].sb_id in source.scheduling_blocks
        events = list(source)
        assert [e.count for e in events] == [0, 1, 2]
        assert [e.index.event_id for e in events] == [0, 2**63 + 1, 1]
        assert all(isinstance(e, SubarrayEventContainer) for e in events)
        assert events[1].tel[1].dl0.trigger.event_type == EventType.SKY_PEDESTAL
        assert events[0].tel[1].dl0.trigger.event_type == EventType.SUBARRAY
        telescope = events[0].tel[1]
        assert telescope.r0.waveform.shape == (1, 109, 1)
        assert telescope.r1.waveform.shape == (1, 109, 1)
        assert telescope.r1.waveform.dtype == np.float32
        np.testing.assert_allclose(telescope.r1.waveform, telescope.r0.waveform * 0.5352)
        assert telescope.r1.event_time == telescope.dl0.trigger.time
        assert telescope.r1.event_type == EventType.SUBARRAY
        np.testing.assert_array_equal(telescope.r0.waveform[0, :, 0], np.arange(109))
        assert telescope.index.tel_id == 1
        assert telescope.index.obs_id == 5000
        assert telescope.monitoring.pointing.altitude.to_value(u.deg) == pytest.approx(70)
        assert events[0].meta["time_source"] == "gps"
        events[0].tel[1].r0.waveform[:] = -1
        assert telescope.r1.waveform[0, 1, 0] == pytest.approx(0.5352)
        assert events[2].tel[1].r0.waveform[0, 0, 0] == 0
        assert len(list(source)) == 3
        for event in events:
            event.validate()
    assert all(reader.closed for reader in readers)
    with pytest.raises(RuntimeError, match="closed"):
        list(source)


@pytest.mark.parametrize("date,factor", [
    ("2001-01-01", 0.4467), ("2002-01-01", 0.5095),
    ("2003-01-01", 0.5352), ("2004-12-31", 0.4776),
])
def test_year_based_dc_to_pe(date, factor):
    assert _default_dc_to_pe(Time(date, scale="utc")) == factor


@pytest.mark.parametrize("date", ["1995-01-01", "2000-12-31", "2005-01-01", "2011-01-01"])
def test_unsupported_dc_to_pe_year(date):
    with pytest.raises(ValueError, match="configure dc_to_pe explicitly"):
        _default_dc_to_pe(Time(date, scale="utc"))


@pytest.mark.parametrize("from_config", [False, True])
def test_dc_to_pe_override(fake_file, from_config):
    kwargs = {"dc_to_pe": 0.75}
    if from_config:
        kwargs = {"config": Config({"WhippleEventSource": kwargs})}
    with WhippleEventSource(fake_file[0], **kwargs) as source:
        for event in source:
            tel = event.tel[1]
            np.testing.assert_allclose(tel.r1.waveform, tel.r0.waveform * 0.75)
            assert tel.r1.event_type == tel.dl0.trigger.event_type


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf])
def test_invalid_dc_to_pe(fake_file, factor):
    with pytest.raises(TraitError, match="finite and positive"):
        WhippleEventSource(fake_file[0], dc_to_pe=factor)


def test_dc_to_pe_run_date(fake_file, records):
    records[0]["nominal_mjd_start"] = Time("2003-09-18").mjd
    with WhippleEventSource(fake_file[0]) as source:
        assert source.dc_to_pe == 0.5352


def test_dc_to_pe_event_date_fallback(fake_file, records):
    records[0]["nominal_mjd_start"] = 0
    records[3]["gps_mjd"] = Time("2004-09-18").mjd
    with WhippleEventSource(fake_file[0]) as source:
        assert source.dc_to_pe == 0.4776


def test_dc_to_pe_unknown_date(fake_file, records):
    records[0]["nominal_mjd_start"] = 0
    records[3].update(gps_is_good=False, record_time_mjd=0)
    with pytest.raises(ValueError, match="configure dc_to_pe explicitly"):
        WhippleEventSource(fake_file[0])


def test_dc_to_pe_unknown_date_can_be_configured(fake_file, records):
    records[0]["nominal_mjd_start"] = 0
    records[3].update(gps_is_good=False, record_time_mjd=0)
    with WhippleEventSource(fake_file[0], dc_to_pe=1.0) as source:
        assert source.dc_to_pe == 1.0


def test_max_events_and_close(fake_file):
    path, readers = fake_file
    source = WhippleEventSource(path, max_events=1)
    assert len(list(source)) == 1
    iterator = iter(source)
    next(iterator)
    source.close()
    assert all(reader.closed for reader in readers)
    assert list(iterator) == []


@pytest.mark.parametrize("allowed,expected", [(None, 3), ({1}, 3), ({2}, 0), (set(), 0)])
def test_allowed_tels(fake_file, allowed, expected):
    with WhippleEventSource(fake_file[0], allowed_tels=allowed) as source:
        assert len(list(source)) == expected


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        WhippleEventSource(tmp_path / "missing.fz")


def test_empty_file(fake_file, records):
    records.clear()
    with pytest.raises(ValueError, match="no decoded event"):
        WhippleEventSource(fake_file[0])
    assert all(reader.closed for reader in fake_file[1])


def test_malformed_adc(fake_file, records):
    records[3]["adc_values"] = [1, 2]
    with WhippleEventSource(fake_file[0]) as source:
        with pytest.raises(ValueError, match="ADC values"):
            list(source)
    assert all(reader.closed for reader in fake_file[1])


def test_changing_run(fake_file, records):
    records[-1]["run_num"] = 5001
    with WhippleEventSource(fake_file[0]) as source:
        with pytest.raises(ValueError, match="Run number or camera"):
            list(source)


def test_time_precision_and_fallback(records):
    record = records[3]
    time, origin = _event_time(record)
    assert origin == "gps"
    assert (time - Time(52800.5, format="mjd", scale="utc")).to_value(u.s) == pytest.approx(0.123456789, abs=1e-9)
    record["gps_is_good"] = False
    time, origin = _event_time(record)
    assert origin == "record"
    assert time.mjd == 52800.5
    record["record_time_mjd"] = 0
    time, origin = _event_time(record)
    assert origin == "unknown"
    assert time == NAN_TIME


@pytest.mark.parametrize("suffix", [".fz", ".fz.gz", ".fz.bz2", ".fz.xz", ".fz.Z", ".fzg", ".fzz"])
def test_compatible_extensions(tmp_path, suffix):
    path = tmp_path / f"gt005000{suffix}"
    assert not WhippleEventSource.is_compatible(path)
    path.touch()
    assert WhippleEventSource.is_compatible(path)
    other = tmp_path / "unrelated.fits"
    other.touch()
    assert not WhippleEventSource.is_compatible(other)


def test_plugin_entry_point():
    plugins = entry_points(group="ctapipe_io")
    assert plugins["WhippleEventSource"].load() is WhippleEventSource


def test_factory(fake_file):
    with EventSource(fake_file[0], max_events=1) as source:
        assert isinstance(source, WhippleEventSource)
        assert len(list(source)) == 1


def test_local_data():
    path = "tests/resources/gt027460.fz.xz"

    with EventSource(path, max_events=5) as source:
        assert isinstance(source, WhippleEventSource)
        events = list(source)
        assert events
        geometry = source.subarray.tel[1].camera.geometry
        for event in events:
            event.validate()
            assert event.tel[1].r0.waveform.shape == (1, geometry.n_pixels, 1)
            np.testing.assert_allclose(
                event.tel[1].r1.waveform,
                event.tel[1].r0.waveform * source.dc_to_pe,
                rtol=1e-6,
            )
            assert event.index.obs_id in source.obs_ids
        # Independently compare the first event to pyfzreader's decoded ADCs.
        with fzreader.FZReader(str(path)) as reader:
            first = next(r for r in reader if r.get("record_was_decoded") and r["record_type"] in ("event", "frame"))
        np.testing.assert_array_equal(events[0].tel[1].r0.waveform[0, :, 0], first["adc_values"][:geometry.n_pixels])
