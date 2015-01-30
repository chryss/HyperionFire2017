#!/usr/bin/env python
# coding: utf-8

## Hyperion fire for AGU - December 2014

# This notebook contains the code to generate the output used in my AGU poster 2014.

### 1. Load prerequisites, filepaths, libraries

#### Imports

import os.path
import random
import json
import h5py
import pickle
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fiona
from shapely.geometry import Point, MultiPoint, Polygon, mapping
from shapely.ops import cascaded_union
from descartes.patch import PolygonPatch
from pygaarst import raster
from pygaarst import landsatutils as l
from pygaarst import hyperionutils as hyp
from pygaarst import irutils as ir
from pygaarst import rasterhelpers as rh
from pygaarst import geomutils as gu
import brewer2mpl    # better color schemes


fireyear = 2004
bandnum = 210
radthresh = 5.0  # Radiance threshold. In the SWIR, non-fire emitted and reflected radiances are much smaller. 
buffrad = 600 # in m

def xy2ji(xytup, geotiff):
    restup = geotiff.xy2ij(*xytup)
    return (restup[1], restup[0])
    
# Directory and file names.

basedir = "/Volumes/SCIENCE/Fire/DATA_BY_PROJECT/HyspIRI_Prep_2014/"
outdir = "/Volumes/SCIENCE/Fire/DATA_BY_PROJECT/HyspIRI_Prep_2014/visual/tmp/"
outgis = "GIS"
dir_2004 = "2004Boundary"
dir_2009 = "2009WoodRiver"
hypscene_2004 = "EO1H0690142004201110PX"
hypscene_2009 = "EO1H0690142009214110KF"

# Select year to work on:

# In[4]:

if fireyear == 2004:
    hypsc = raster.Hyperionscene(os.path.join(basedir, dir_2004, hypscene_2004))
    hypsc.infix = '_CLIP'
    filedir = dir_2004
    firename = 'boundary'
    figdim = (15, 19)
    colbarpos = [0.125, 0.2, 0.725, 0.025]
elif fireyear == 2009:
    hypsc = raster.Hyperionscene(os.path.join(basedir, dir_2009, hypscene_2009))
    hypsc.infix = '_CLIP'
    filedir = dir_2009
    firename = 'woodriver'
    figdim = (15, 25)
    colbarpos = [0.125, 0.125, 0.74, 0.02]
else:
    raise Exception("Fireyear not implemented.")


b220 = hypsc.band220
b210 = hypsc.band210

if bandnum == 210:
    bd = b210
elif bandnum == 220:
    bd = b220
else:
    print("Please choose suitable band number")

i_fire, j_fire = np.where(bd.radiance > radthresh)
e_fire, n_fire = bd.ij2xy(i_fire, j_fire)
# the fire points are in the center of the pixels, while the easting and northing designate the top-left corner
e_fire += 0.5 * bd.delx
n_fire += 0.5 * bd.dely   # note that dely is negative
# Easting and northing as x and y coordinate mesh

# calculate fire shapes

firecircles = [Point(eastx, northy).buffer(buffrad) for eastx, northy in zip(e_fire, n_fire)]
circbuffer = cascaded_union(firecircles)

firemask = np.zeros(bd.data.shape)
firemask[bd.radiance > radthresh] = 1. 

circbuffer_ij = cascaded_union(
    [ Polygon([xy2ji(tup, bd) for tup in list(poly.exterior.coords)])
         for poly in circbuffer])

circbuffer_mask = gu.overlayvectors(bd.data, circbuffer_ij)
circbuffer_mask -= firemask

#### Buffer pixels
# Calculate spectra and save to file

i_buffer, j_buffer = np.where(circbuffer_mask == 1)
specs = [hypsc.spectrum(i, j) for i, j in itertools.izip(i_buffer, j_buffer)]
outfn = os.path.join(outdir, "%s_circbufferspecs_calib.hdf5" % firename)

rh.save_hypspec_to_hdf5(outfn, hypsc, specs, i_buffer, j_buffer)
