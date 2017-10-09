#!/usr/local/env python

from __future__ import print_function, division
import os.path
from sklearn.cluster import MiniBatchKMeans
import sklearn
import numpy
import scipy
from osgeo import gdal

gdal.UseExceptions()

print ("scikit-learn verion: {0}".format(sklearn.__version__))
print ("numpy version: {0}".format(numpy.__version__))
print ("scipy version: {0}".format(scipy.__version__))
print ("GDAL version: {0}".format(gdal.__version__))

hypfile_2004 = "/Volumes/SCIENCE/Fire/DATA_BY_PROJECT/HyspIRI_Prep_2014/2004Boundary/EO1H0690142004201110PX/EO1H0690142004201110PX_B220_L1GST_CLIP.TIF"
dataobj = gdal.Open(hypfile_2004)

data = dataobj.ReadAsArray()
print (data.dtype)
print(os.path.splitext(__file__))