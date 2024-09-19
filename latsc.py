import logging
import numpy
import astropy.units as u
import astropy.io.fits as pyfits

from dataclasses import dataclass
from scipy.interpolate import CubicSpline
from astropy.time import Time
from astropy.coordinates import TEME, ITRS, GCRS, CartesianRepresentation
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator


@dataclass
class SpacecraftData:
    """
    Class to hold the LAT spacecraft position data
    """
    time: Time
    loc: EarthLocation
    teme: TEME

    def teme2itrs(self, obstime):
        """
        Convert the True Equator, Mean Equinox frame (TEME)
        data into International Terrestrial Reference System (ITRS)
        
        Parameters
        ----------
        obstime: Time
            Observation time to assume (to account for Earth rotation)

        Returns
        -------
        EarthLocationl
            Corresponding ITRS position
        """
        itrs_geo = self.teme.transform_to(ITRS(obstime=obstime))
        return itrs_geo.earth_location


class LatSpacecraft:
    """
    Class to compute the LAT spacecraft zenith direction
    """
    data = None

    def __init__(self, sc_file, step=1, log=None):
        """
        Constructor. Loads the spacecraft file and 
        creates it's trajectory interpolators.

        Parameters
        ----------
        sc_file: str
            LAT spacecraft file name
        step: int
            Step with which to load the position data
            from the spacecraft file.
        log: logging.Logger
            Log to use. If None, a default log will be created.
        """

        if log is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = log.getChild(__name__)
        
        self.data = self.load_sc_data(sc_file, step)
        self._data_sc_zenith = self.get_sc_zenith(self.data).icrs

    def load_sc_data(self, sc_file, step):
        """
        Loads LAT spacecraft position data from the specified file.

        Parameters
        ----------
        sc_file: str
            LAT spacecraft file name
        step: int
            Step with which to load the position data
            from the spacecraft file.

        Returns
        -------
        sc: SpacecraftData
            Spacecraft position data
        """
        
        self.log.debug('[load_sc_data] reading %s', sc_file)
        with pyfits.open(sc_file) as hdus:
            met0 = Time(
                hdus['Primary'].header['MJDREFI'] + hdus['Primary'].header['MJDREFF'],
                format='mjd',
                scale=hdus['Primary'].header['TIMESYS'].lower()
            )

            start = Time(
                met0.gps + hdus['SC_DATA'].data['START'][::step],
                format='gps',
                scale='utc'
            )
            geo_lat = hdus['SC_DATA'].data['LAT_GEO'][::step] * u.deg
            geo_lon = hdus['SC_DATA'].data['LON_GEO'][::step] * u.deg
            geo_rad = hdus['SC_DATA'].data['RAD_GEO'][::step] * u.m
            sc_position = hdus['SC_DATA'].data['SC_POSITION'][::step] * u.m

        self.log.debug('[load_sc_data] computing earth location')
        # obsloc = EarthLocation(
        #     lat=geo_lat,
        #     lon=geo_lon,
        #     height=geo_rad
        # )
        # https://stackoverflow.com/questions/46433702/how-to-convert-earth-centered-inertial-eci-coordinates-to-earth-centered-earth
        cartrep = CartesianRepresentation(
            sc_position.transpose()
        )
        gcrs = GCRS(cartrep, obstime=start)
        itrs = gcrs.transform_to(ITRS(obstime=start))
        obsloc = itrs.earth_location

        self.log.debug('[load_sc_data] preparing TEME representation')
        # Based on https://docs.astropy.org/en/stable/coordinates/satellites.html
        sc_teme = TEME(
            CartesianRepresentation(sc_position.transpose()), 
            obstime=start
        )

        self.log.debug('[load_sc_data] done')
        return SpacecraftData(start, obsloc, sc_teme)

    @staticmethod
    def get_sc_zenith(sc):
        obsframe = AltAz(
            location=sc.loc,
            # location=sc.teme2itrs(sc.time),
            obstime=sc.time
        )
        
        sc_zenith = SkyCoord(
            alt=numpy.repeat(90, len(sc.time)),
            az=numpy.repeat(0, len(sc.time)),
            unit='deg',
            frame=obsframe
        )
        return sc_zenith
    
    def sc_zenith(self, obstime):
        zenith = SkyCoord(
            ra = numpy.interp(obstime.gps, self.data.time.gps, self._data_sc_zenith.ra.deg),
            dec = numpy.interp(obstime.gps, self.data.time.gps, self._data_sc_zenith.dec.deg),
            unit='deg',
            frame='icrs'
        )
        return zenith
