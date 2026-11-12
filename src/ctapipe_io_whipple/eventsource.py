"""Read local GDF/ZEBRA files using pyfzreader (imported as ``fzreader``)."""

from pathlib import Path

import astropy.units as u
import fzreader
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time, TimeDelta
from traitlets import TraitError, validate
from ctapipe.containers import (
    NAN_TIME,
    EventType,
    ObservationBlockContainer,
    PixelStatus,
    SchedulingBlockContainer,
    SubarrayDL0Container,
    SubarrayEventContainer,
    SubarrayEventIndexContainer,
    SubarrayMonitoringContainer,
    SubarrayPointingContainer,
    SubarrayTriggerContainer,
    TelescopeDL0Container,
    TelescopeEventContainer,
    TelescopeEventIndexContainer,
    TelescopeMonitoringContainer,
    TelescopePointingContainer,
    TelescopeR0Container,
    TelescopeR1Container,
    TelescopeTriggerContainer,
)
from ctapipe.coordinates import TelescopeFrame, CameraFrame
from ctapipe.core import Map
from ctapipe.core.traits import Float
from ctapipe.instrument import (
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    OpticsDescription,
    PixelShape,
    SubarrayDescription,
    TelescopeDescription,
)
from ctapipe.io import DataLevel, EventSource

__all__ = ["WhippleEventSource"]

# Kildea et al. (2007), doi:10.1016/j.astropartphys.2007.05.004.
FOCAL_LENGTH = 7.3 * u.m
LOCATION = EarthLocation.from_geodetic(
    "-110d52m45.1s",
    "31d40m49.7s",
    2300 * u.m,
)
TEL_ID = 1


def _is_event(record):
    return record.get("record_was_decoded", False) and record.get("record_type") in (
        "event", "frame"
    )


def create_camera_geometry(nadc):
    """Build a camera from the positions, circular radii and neighbors in pyfzreader.

    Keep the JSON coordinates in degrees, with x as fov_lon and y as fov_lat.
    In TelescopeFrame these axes point toward increasing azimuth and altitude,
    respectively. Unlike CameraFrame, the horizontal coordinate comes first.
    The JSON exporter and WhippleCams.h do not specify the absolute sky
    orientation: this direct mapping preserves the supplied layout but its
    rotation/reflection relative to the sky still needs independent validation.
    ADC channels beyond ``npix`` are not camera pixels.
    """
    camera = fzreader.get_camera_geometry_by_nadc(nadc)
    if camera is None or camera["nadc"] != nadc:
        raise ValueError(f"Unsupported Whipple camera with {nadc} ADC channels")
    return CameraGeometry(
        name=f"Whipple{camera['npix']}",
        pix_id=np.arange(camera["npix"]),
        pix_x=np.asarray(camera["x"]) * u.deg,
        pix_y=np.asarray(camera["y"]) * u.deg,
        pix_area=np.pi * (np.asarray(camera["r"]) * u.deg) ** 2,
        pix_type=PixelShape.CIRCLE,
        neighbors=camera["neighbors"],
        frame=TelescopeFrame(),
    ).transform_to(CameraFrame(focal_length=FOCAL_LENGTH))


def _create_subarray(nadc):
    geometry = create_camera_geometry(nadc)
    optics = OpticsDescription(
        name="Whipple10m",
        size_type="MST",
        reflector_shape="DAVIES_COTTON",
        n_mirrors=1,
        equivalent_focal_length=FOCAL_LENGTH,
        effective_focal_length=np.nan * u.m,
        mirror_area=75 * u.m**2,
        n_mirror_tiles=248,
    )
    # Dummy readout metadata lets ctapipe image extractors run. The sampling
    # rate and pulse shape are not measurements of Whipple electronics.
    readout = CameraReadout(
        name=geometry.name,
        sampling_rate=1 * u.GHz,
        reference_pulse_shape=np.ones((1, 1)),
        reference_pulse_sample_width=1 * u.ns,
        n_channels=1,
        n_pixels=geometry.n_pixels,
        n_samples=1,
    )
    camera = CameraDescription(geometry.name, geometry, readout=readout)
    telescope = TelescopeDescription("Whipple10m", optics, camera)
    return SubarrayDescription(
        "Whipple", {TEL_ID: np.zeros(3) * u.m}, {TEL_ID: telescope}, LOCATION
    )


def _event_time(record):
    """Use decoded UTC GPS time, with explicitly marked DAQ-time fallback."""
    if record.get("gps_is_good", False) and record.get("gps_mjd", 0) > 0:
        # Add SI seconds to UTC midnight, also on days with a leap second.
        midnight = Time(record["gps_mjd"], format="mjd", scale="utc")
        delta = TimeDelta(
            record["gps_utc_sec"], record.get("gps_ns", 0) * 1e-9, format="sec"
        )
        return midnight + delta, "gps"
    mjd = record.get("record_time_mjd", 0)
    if np.isfinite(mjd) and mjd > 0:
        return Time(mjd, format="mjd", scale="utc"), "record"
    # Use ctapipe's standard unknown timestamp sentinel.
    return NAN_TIME, "unknown"


def _default_dc_to_pe(time):
    """Return the PE/DC multiplier for a supported UTC observation year."""
    if NAN_TIME == time:
        raise ValueError(
            "Cannot determine dc_to_pe without an observation year; "
            "configure dc_to_pe explicitly"
        )
    year = int(time.utc.ymdhms.year)
    factors = {2001: 0.4467, 2002: 0.5095, 2003: 0.5352, 2004: 0.4776}
    try:
        return factors[year]
    except KeyError as error:
        raise ValueError(
            f"No default dc_to_pe is defined for {year}; "
            "configure dc_to_pe explicitly"
        ) from error


class WhippleEventSource(EventSource):
    """Read one local Whipple run, including sky and pedestal events.

    Integrated, uncalibrated ADC charges are exposed as R0 waveforms with shape
    ``(1, n_pixels, 1)``. R1 applies a scalar DC-to-PE conversion, without
    pedestal subtraction or relative pixel gain corrections.
    Older pedestal frames have an independent counter: their event IDs are
    ``2**63 + frame_num`` to avoid collisions with ordinary event numbers.
    Telescope ID is always 1. Files can be iterated repeatedly until closed.
    """

    dc_to_pe = Float(
        default_value=None,
        allow_none=True,
        help=(
            "Photoelectrons per digital count. None selects the value for the "
            "observation year; unsupported or unknown years require an explicit value."
        ),
    ).tag(config=True)

    @validate("dc_to_pe")
    def _validate_dc_to_pe(self, proposal):
        value = proposal["value"]
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise TraitError("dc_to_pe must be finite and positive, or None")
        return value

    def __init__(self, input_url=None, config=None, parent=None, **kwargs):
        super().__init__(input_url=input_url, config=config, parent=parent, **kwargs)
        self._reader = None
        self._closed = False
        if self.input_url is None or not self.input_url.is_file():
            raise FileNotFoundError(f"Not a local Whipple file: {self.input_url}")

        self.run_header = {}
        with fzreader.FZReader(str(self.input_url)) as reader:
            for record in reader:
                if record.get("record_type") == "run" and record.get("record_was_decoded"):
                    self.run_header = record.copy()
                if _is_event(record):
                    self._obs_id = int(record["run_num"])
                    self._nadc = int(record["nadc"])
                    break
            else:
                raise ValueError("Whipple file contains no decoded event or pedestal frame")

        if self.run_header and self.run_header["run_num"] != self._obs_id:
            raise ValueError("Run header and first event have different run numbers")
        self._subarray = _create_subarray(self._nadc)
        start = self.run_header.get("nominal_mjd_start", 0)
        scheduled_start_time = NAN_TIME
        if np.isfinite(start) and start > 0:
            scheduled_start_time = Time(start, format="mjd", scale="utc")
        if self.dc_to_pe is None:
            calibration_time = scheduled_start_time
            if NAN_TIME == calibration_time:
                calibration_time, _ = _event_time(record)
            self.dc_to_pe = _default_dc_to_pe(calibration_time)
        self._observation_blocks = {
            self._obs_id: ObservationBlockContainer(
                obs_id=self._obs_id,
                sb_id=np.uint64(self._obs_id),
                producer_id="Whipple",
                scheduled_start_time=scheduled_start_time,
            )
        }
        self._scheduling_blocks = {
            self._obs_id: SchedulingBlockContainer(
                sb_id=np.uint64(self._obs_id), producer_id="Whipple"
            )
        }

    @staticmethod
    def is_compatible(file_path):
        """Recognize local Whipple filename conventions without reading the run."""
        path = Path(file_path)
        return path.is_file() and path.name.endswith(
            (".fz", ".fz.gz", ".fz.bz2", ".fz.xz", ".fz.Z", ".fzg", ".fzz")
        )

    @property
    def subarray(self):
        return self._subarray

    @property
    def is_simulation(self):
        return False

    @property
    def datalevels(self):
        return (DataLevel.R0, DataLevel.R1)

    @property
    def observation_blocks(self):
        return self._observation_blocks

    @property
    def scheduling_blocks(self):
        return self._scheduling_blocks

    def _generator(self):
        if self._closed:
            raise RuntimeError("WhippleEventSource is closed")
        if self._reader is not None:
            raise RuntimeError("Concurrent iteration of a WhippleEventSource is not supported")
        if self.allowed_tels is not None and TEL_ID not in self.allowed_tels:
            return
        tracking = None
        count = 0
        with fzreader.FZReader(str(self.input_url)) as reader:
            self._reader = reader
            try:
                for record in reader:
                    if self._closed:
                        return
                    if not record.get("record_was_decoded", False):
                        continue
                    if record["record_type"] == "tracking":
                        tracking = record
                    elif _is_event(record):
                        yield self._make_event(record, count, tracking)
                        count += 1
                        if self._closed or (self.max_events and count >= self.max_events):
                            return
            finally:
                self._reader = None

    def _make_event(self, record, count, tracking):
        if record["run_num"] != self._obs_id or record["nadc"] != self._nadc:
            raise ValueError("Run number or camera changes within the Whipple file")
        adc = np.asarray(record["adc_values"], dtype=np.int16)
        if adc.shape != (self._nadc,):
            raise ValueError(f"Expected {self._nadc} ADC values, got shape {adc.shape}")
        is_frame = record["record_type"] == "frame"
        event_id = (2**63 + int(record["frame_num"])) if is_frame else int(record["event_num"])
        event_type = {
            "sky": EventType.SUBARRAY, "pedestal": EventType.SKY_PEDESTAL
        }.get(record.get("event_type"), EventType.UNKNOWN)
        time, time_source = _event_time(record)
        az = alt = np.nan * u.rad
        telescope_pointing = None
        if tracking is not None:
            az = (tracking["telescope_az_deg"] * u.deg).to(u.rad)
            alt = (tracking["telescope_el_deg"] * u.deg).to(u.rad)
            telescope_pointing = TelescopePointingContainer(azimuth=az, altitude=alt)

        npix = self.subarray.tel[TEL_ID].camera.geometry.n_pixels
        telescope = TelescopeEventContainer(
            count=count,
            index=TelescopeEventIndexContainer(
                obs_id=self._obs_id, event_id=event_id, tel_id=TEL_ID,
            ),
            r0=TelescopeR0Container(waveform=adc[:npix].copy()[None, :, None]),
            r1=TelescopeR1Container(
                waveform=(adc[:npix].astype(np.float32) * self.dc_to_pe)[None, :, None],
                event_type=event_type,
                event_time=time,
                pixel_status=np.full(npix, PixelStatus.HIGH_GAIN_STORED, dtype=np.uint8),
            ),
            dl0=TelescopeDL0Container(
                trigger=TelescopeTriggerContainer(time=time, event_type=event_type),
            ),
            monitoring=TelescopeMonitoringContainer(pointing=telescope_pointing),
        )
        event = SubarrayEventContainer(
            count=count,
            index=SubarrayEventIndexContainer(obs_id=self._obs_id, event_id=event_id),
            dl0=SubarrayDL0Container(
                trigger=SubarrayTriggerContainer(
                    time=time,
                    event_type=event_type,
                    tels_with_trigger=np.array([TEL_ID], dtype=np.uint16),
                    tels_with_data=np.array([TEL_ID], dtype=np.uint16),
                ),
            ),
            monitoring=SubarrayMonitoringContainer(
                pointing=SubarrayPointingContainer(azimuth=az, altitude=alt),
            ),
            tel=Map(TelescopeEventContainer, {TEL_ID: telescope}),
        )
        event.meta.update(
            origin="Whipple", time_source=time_source, record_type=record["record_type"],
        )
        return event

    def close(self):
        self._closed = True
        if self._reader is not None:
            self._reader.__exit__(None, None, None)
            self._reader = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
