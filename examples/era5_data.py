import pygrib

grib = 'era5.grib' # Set the file name of your input GRIB file
grbs = pygrib.open(grib)
 
grb = grbs.select()[0]
data = grb.values

import ipdb; ipdb.set_trace()