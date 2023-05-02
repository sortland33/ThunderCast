#Author: John L. Cintineo
import numpy as np

def solar_angles(dt,lon,lat):
  '''
  ---------------------------------------------------------------------
   Compute solar angles

   input:
         dt   = python datetime object
         ###jday = julian day
         ###tu   = time of day - fractional hours
         lon  = latitude in degrees
         lat  = latitude in degrees

    output:
         asol = solar zenith angle in degrees
         phis = solar azimuth angle in degrees

  ----------------------------------------------------------------------
  '''

  if(not(isinstance(dt,list)) and not(isinstance(dt,np.ndarray))): dt = [dt]
  lon = np.array(lon)
  lat = np.array(lat)

  hour = np.array([_.hour for _ in dt])
  minute = np.array([_.minute for _ in dt])
  jday = np.array([int(_.strftime('%j')) for _ in dt])

  dtor = np.pi/180.


  tu = hour + minute/60.0

  #      mean solar time
  tsm = tu + lon/15.0
  xlo = lon*dtor
  xla = lat*dtor
  xj = np.copy(jday)

  #      time equation (mn.dec)
  a1 = (1.00554*xj - 6.28306)  * dtor
  a2 = (1.93946*xj + 23.35089) * dtor
  et = -7.67825*np.sin(a1) - 10.09176*np.sin(a2)

  #      true solar time
  tsv = tsm + et/60.0
  tsv = tsv - 12.0

  #      hour angle
  ah = tsv*15.0*dtor

  #      solar declination (in radian)
  a3 = (0.9683*xj - 78.00878) * dtor
  delta = 23.4856*np.sin(a3)*dtor

  #     elevation, azimuth
  cos_delta = np.cos(delta)
  sin_delta = np.sin(delta)
  cos_ah = np.cos(ah)
  sin_xla = np.sin(xla)
  cos_xla = np.cos(xla)

  amuzero = sin_xla*sin_delta + cos_xla*cos_delta*cos_ah
  elev = np.arcsin(amuzero)
  cos_elev = np.cos(elev)
  az = cos_delta*np.sin(ah)/cos_elev
  caz = (-cos_xla*sin_delta + sin_xla*cos_delta*cos_ah) / cos_elev
  azim=0.*az
  index=np.where(az >= 1.0)
  if np.size(index) > 0 : azim[index] = np.arcsin(1.0)
  index=np.where(az <= -1.0)
  if np.size(index) > 0 : azim[index] = np.arcsin(-1.0)
  index=np.where((az > -1.0) & (az < 1.0))
  if np.size(index) > 0 : azim[index] = np.arcsin(az[index])

  index=np.where(caz <= 0.0)
  if np.size(index) > 0 : azim[index] = np.pi - azim[index]

  index=np.where((caz > 0.0) & (az <= 0.0))
  if np.size(index) > 0 : azim[index] = 2 * np.pi + azim[index]
  azim = azim + np.pi
  pi2 = 2 * np.pi
  index=np.where(azim > pi2)
  if np.size(index) > 0 : azim[index] = azim[index] - pi2

  #     conversion in degrees
  elev = elev / dtor
  asol = 90.0 - elev
  phis = azim / dtor
  sol_zenith=asol
  sol_azimuth=phis

  return sol_zenith, sol_azimuth

