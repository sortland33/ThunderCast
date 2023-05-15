import numpy as np
import sys
def earth_to_fgf(\
                lat,\
                lon,\
                nav_type='NOAA',\
                sub_lon_grid=-75.,\
                sat_height=42164.,\
                cfac=0.000056,\
                lfac=-0.000056,\
                ccoff=-0.101332,\
                lloff=0.128212,\
                pheight=0,\
                cflip=False,\
                lflip=False):

  #for 10km, -75W cfac = 0.00028
  #               lfac = -0.00028
  #               ccoff = -0.101220
  #               lloff = 0.128100
  #for 2km, -75W  cfac = 0.000056
  #               lfac = -0.000056
  #               ccoff = -0.101332
  #               lloff = 0.128212
  # See http://data.eol.ucar.edu/datafile/nph-get/558.003/GOES-R_Users_Guide_V4.pdf
  # for more fixed grid stuff (page 48).

#needs to retrun col an row
  #sat height in km from earth's center
  #pheight = parallax height in km

  DTOR = 0.0174532925199433 #degrees to radians
  R_EQ = 6378.169      #km
  R_POL= 6356.5838     #km

  if(isinstance(lat,(int,float))):
    lat = np.array([lat])
  elif(isinstance(lat,list)):
    lat = np.array(lat)
  if(isinstance(lon,(int,float))):
    lon = np.array([lon])
  elif(isinstance(lon,list)):
    lon = np.array(lon)


  if(np.size(lat) == 1):
    dimen = 1    
  else:
    dimen = np.shape(lat)

  #initalize row/column
  col = np.zeros((dimen)) - 999.
  row = np.zeros((dimen)) - 999.

  if (lat.min() < -90 or lat.max() > 90 or lon.min() < -180 or lon.max() > 360):
    print(lat.min(),lat.max(),lon.min(),lon.max())
    print('Invalid lat/lon input')
    return np.array([-1]),np.array([-1])
 

  nav_type = nav_type.upper()

  geographic_lat = lat * DTOR
  geographic_lon = lon * DTOR
  sub_lon = sub_lon_grid * DTOR
  
  geocentric_lat = np.arctan(((R_POL*R_POL)/(R_EQ*R_EQ))*np.tan(geographic_lat))
  
  r_earth = R_POL/np.sqrt(1.0 -((R_EQ*R_EQ - R_POL*R_POL)/(R_EQ*R_EQ))*np.cos(geocentric_lat)*np.cos(geocentric_lat))

  rl = r_earth + pheight #parralax correction
  r_1 = sat_height - rl*np.cos(geocentric_lat)*np.cos(geographic_lon - sub_lon)
  r_2 = -rl*np.cos(geocentric_lat)*np.sin(geographic_lon - sub_lon)
  r_3 = rl*np.sin(geocentric_lat)
    
  #initialize intermedate cooridinate
  #lamda = -999.0
  #theta = -999.0

  #if (r_1 gt sat_height) then return,0
  
  #   check for visibility, whether the point on the Earth given by the
  #   latitude/longitude pair is visible from the satellte or not. This 
  #   is given by the dot product between the vectors of:
  #   1) the point to the spacecraft,
  #   2) the point to the centre of the Earth.
  #   If the dot product is positive the point is visible otherwise it is invisible.
  # Taken from EUMETSAT/JMA CGMS code

  pow_eq = (R_EQ/R_POL) ** (2.0)
  dotprod = r_1*(rl * np.cos(geocentric_lat) * np.cos(geographic_lon - sub_lon)) - (r_2*r_2) - ((r_3*r_3)*(pow_eq))
   
  #if (dotprod le 0.0) then return,0

  igood = np.where((r_1 <= sat_height) & (dotprod > 0.0))

  if (len(igood[0]) > 0):

    if (nav_type == 'NOAA'):

      #zero-based
      lamda = np.arcsin(-r_2[igood]/np.sqrt(r_1[igood]*r_1[igood] + r_2[igood]*r_2[igood] + r_3[igood]*r_3[igood]))
      theta = np.arctan(r_3[igood]/r_1[igood])
      col[igood] = (lamda - ccoff)/cfac
      row[igood] = (theta - lloff)/lfac
  
      #convert to one-based
      col[igood] = col[igood] + 1.0
      row[igood] = row[igood] + 1.0

    elif (nav_type == 'EUMETSAT'):

      #one-based
      lamda = np.arctan(-r_2[igood]/r_1[igood])
      theta = np.arcsin(-r_3[igood]/np.sqrt(r_1[igood]*r_1[igood] + r_2[igood]*r_2[igood] + r_3[igood]*r_3[igood]))

      col[igood] = ccoff + (2**(-16.) * lamda * cfac)
      row[igood] = lloff + (2**(-16.) * theta * lfac)

    elif (nav_type == 'JMA'):
  
      #one-based
      lamda = np.arctan(-r_2[igood]/r_1[igood])
      theta = np.arcsin(-r_3[igood]/np.sqrt(r_1[igood]*r_1[igood] + r_2[igood]*r_2[igood] + r_3[igood]*r_3[igood]))
  
      col[igood] = ccoff + (2**(-16.) * lamda * cfac / DTOR)
      row[igood] = lloff + (2**(-16.) * theta * lfac / DTOR)

    else:

      print('Invalid nav type: ' + nav_type)
      return np.array([-1]),np.array([-1])


    if (cflip): col[igood] = ccoff*2 - col[igood]
    if (lflip): row[igood] = lloff*2 - row[igood]

  return row, col
