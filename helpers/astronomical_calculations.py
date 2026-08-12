"""
This file contains functions for retrieving information on the sun angles and air mass between sun and the pv panels.
Functions utilize tools from PVlib.

Angle of incidence is the angle between the solar panel normal angle and the angle of sunlight hitting the panel.
Solar azimuth and Zenith are the spherical coordinate angles used for describing the angle of the sun.
Air mass is required by the perez model.

Functions from this file are mainly required for irradiance transpositions.


Terminology:
Zenith(point): point directly above the observer.
Zenith angle(degrees): distance in degrees from the point above.
solar apparent zenith(degrees): apparent distance of the sun from the zenith as observed at the location through the atmosphere.
this is in contrast to solar zenith which is the actual sun position. The difference between these two zenith values is
caused by the atmosphere.
AOI(degrees): angle of incidence. Difference between solar panel normal vector and the vector of incoming sunlight. At AOI of 0,
reflections are minimal and relative surface area is at 1. At 90, panels do not receive direct sunlight and relative
surface area is 0.
Azimuth(degrees): 0 for north, 90 for east, 180 for south, 270 for west.

Author: TimoSalola (Timo Salola).
"""

from datetime import datetime
import pandas
import pvlib.atmosphere
from helpers import config
from pvlib import location, irradiance


def get_solar_angle_of_incidence_fast(dt:datetime)-> float:
    """
    Estimates solar angle of incidence at given datetime. Other parameters, tilt, azimuth and geolocation are read from
    config.py.
    :param dt: Datetime object, should include date and time.
    :return: Angle of incidence in degrees. Angle between sunlight and solar panel normal

    Optimized version, should work well
    """


    solar_azimuth, solar_apparent_zenith = get_solar_azimuth_zenit_fast(dt)
    panel_tilt = config.tilt
    panel_azimuth = config.azimuth

    # angle of incidence, angle between direct sunlight and solar panel normal
    angle_of_incidence = irradiance.aoi(panel_tilt, panel_azimuth, solar_apparent_zenith, solar_azimuth)

    # restricting AOI values as projection functions do not expect AOI higher than 90. Should never be lower than 0 but setting a limit anyways
    angle_of_incidence = angle_of_incidence.clip(lower=0, upper=90)

    return angle_of_incidence



def get_air_mass_fast(time: datetime)-> float:
    """
    Generates value for air mass using pvlib default model(kastenyoung1989).
    This value tells us the relative thickness of atmosphere between sun and the PV panels.
    :param time: python datetime
    :return: air mass value, may return nans if AOI is over 90
    """
    solar_zenith = get_solar_azimuth_zenit_fast(time)[1]
    air_mass = pvlib.atmosphere.get_relative_airmass(solar_zenith)
    return air_mass



def get_solar_azimuth_zenit_fast(dt: datetime)-> (float, float):
    """
    Returns apparent solar zenith and solar azimuth angles in degrees.
    :param dt: time to compute the solar position for.
    :return: azimuth, zenith
    """

    # panel location and installation parameters from config file
    panel_latitude = config.latitude
    panel_longitude = config.longitude

    # panel location object, required by pvlib
    panel_location = location.Location(panel_latitude, panel_longitude, tz=config.timezone)

    # solar position object
    solar_position = panel_location.get_solarposition(dt)

    # apparent zenith and azimuth, Using apparent for zenith as the atmosphere affects sun elevation.
    # apparent_zenith = Sun zenith as seen and observed from earth surface
    # zenith = True Sun zenith, would be observed if Earth had no atmosphere
    solar_apparent_zenith = solar_position["apparent_zenith"]
    solar_azimuth = solar_position["azimuth"]

    return solar_azimuth, solar_apparent_zenith