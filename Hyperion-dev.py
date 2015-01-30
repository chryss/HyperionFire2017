
## Hyperion fire detection - April 2014

### Load prerequisites, filepaths, libraries

# Imports.

# In[1]:

get_ipython().magic(u'pylab inline')
import numpy as np
import h5py
import os.path
import random
import pickle
import json
from itertools import izip_longest
import matplotlib.pyplot as plt
from pygaarst import raster
from pygaarst import landsatutils as l
from pygaarst import hyperionutils as hyp
from pygaarst import irutils as ir
import brewer2mpl
import scipy.stats as stat


# Out[1]:

#     Populating the interactive namespace from numpy and matplotlib
# 

# Set up plot options. 

# In[2]:

bmap = brewer2mpl.get_map('Set1', 'qualitative', 7)
colors = bmap.mpl_colors
mpl.rcParams['axes.color_cycle'] = colors
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'Calibri'})


# For interactive work, Pygaarst libraries need re-importing after editing.

# In[75]:

reload(raster)
reload(ir)
reload(l)
reload(hyp)


# Out[75]:

#     <module 'pygaarst.hyperionutils' from 'pygaarst/hyperionutils.pyc'>

# Directory and file names.

# In[3]:

basedir = "/Volumes/SCIENCE/Fire/DATA_BY_PROJECT/HyspIRI_Prep_2014/"
outdir = "/Volumes/SCIENCE/Fire/DATA_BY_PROJECT/HyspIRI_Prep_2014/visual/tmp/"
dir_2004 = "2004Boundary"
dir_2009 = "2009WoodRiver"
hypscene_2004 = "EO1H0690142004201110PX"
aliscene_2004 = "EO1A0690142004201110PX"
landsatscene_2004 = "LE70690142004201EDC01"
hypscene_2009 = "EO1H0690142009214110KF"
aliscene_2009 = "EO1A0690142009214110KF"
landsatscene_2009 = "LT50680152009215GLC00"  # There is also a simultaneous L7 scene, LE70690152009214EDC00, but it's stripey


# In[4]:

hyp2004 = raster.Hyperionscene(os.path.join(basedir, dir_2004, hypscene_2004))
hyp2004.infix = '_CLIP'
hyp2009 = raster.Hyperionscene(os.path.join(basedir, dir_2009, hypscene_2009))
hyp2009.infix = '_CLIP'


# Select year to work on:

# In[5]:

fireyear = 2004

if fireyear == 2004:
    hypsc = hyp2004
    firename = 'boundary'
elif fireyear == 2009:
    hypsc = hyp2009
    firename = 'woodriver'
else:
    raise Exception("Fireyear not implemented.")


### Helper functions

# Planck spectral radiance (blackbody).

# In[6]:

# SI units except for lambda
h = 6.626068e-34  # Planck's constant, m^2 kg / s
c = 2.99792e8 # speed of light, m / s
kB = 1.38065e-23 # Boltzmann's constant, m^2 kg / s^2 / K

def specrad(lamb, T):
	# blackbody radiator radiance in W/m^2/um/sr; T in K; lambda in micrometres
    lamb = lamb * 1.0e-6 # convert from micrometres to metres
    rad = 1.0e-6 * ( 2*h*c**2 ) / ( lamb**5*( np.exp( (h*c)/(lamb*kB*T) ) - 1 ) )
    return rad


# Find nearest Hyperion wavelength.

# In[7]:

def find_nearest_hyp(wavelength):
    """
    Returns index and wavelength of Hyperion band closest to input wavelength
    
    Arguments:
      wavelength (float): wavelength in nm
      
    Returns:
      idx (int): band index of closest band, starting at 0
      band (str): band name of closest band, starting at 'band1'
      bandwavelength (float): closest band wavelength in nm
    """
    bands = hyp.gethyperionbands().Hyperion_Band
    wavs = hyp.gethyperionbands().Average_Wavelength_nm
    idx = (np.abs(wavs - wavelength)).argmin()
    return idx, bands[idx], wavs[idx]


# Save a set of spectra to a HDF5 file.

# In[44]:

def save_to_hdf5(outfn, hypsc, spectra, i_coord, j_coord):
    """
    Save a set of spectra to HDF5
    
    Arguments:
      outfn (str): file path of the HDF5 file. Will overwrite.
      hypsc (HyperionScene): The Hyperion scene from which the spectra are loaded
      spectra (float): Nympy array. num coord by num wavelengths
      i_coord (int): pixel row coordinate array
      j_coord (int): pixel column coordinate array
    """
    specs_arr = np.array(specs)
    with h5py.File(outfn, 'w') as fh:
        rowidx = fh.create_dataset("i_row_idx", data=i_coord)
        colidx = fh.create_dataset("j_col_idx", data=j_coord)
        spec = fh.create_dataset("spectrum", data=spectra)
        bandnames = fh.create_dataset("bandname", data=hypsc.hyperionbands[hypsc.band_is_calibrated])
        bandidx = fh.create_dataset("bandindex", data=np.where(hypsc.band_is_calibrated)[0])
        bandwavelength = fh.create_dataset("bandwavelenght_nm", data=hypsc.calibratedwavelength_nm)


### Load data: radiance, fire and buffer candidate pixels

# Sample band to use for rough fire detection

# In[8]:

b220 = hypsc.band220


# In[9]:

b220.radiance.dtype


# Out[9]:

#     dtype('float32')

# In[12]:

saveout = False
f, ax = plt.subplots(figsize=(15, 25))
#output = ax.imshow(b220.radiance, cmap='hot', clim=(0, 70))
output = ax.pcolor(np.flipud(b220.radiance), cmap='hot', vmin=0, vmax=70)
ax.set_title("%s %s fire Band 220, Radiance in $W / (m^2 \mu m\, sr)$" % (fireyear, firename.capitalize()))
ax.set_aspect('equal')
ax.set_axis_off()
if fireyear == 2004:
    position=f.add_axes([0.05, 0.2, 0.85, 0.025])
else:
    position=f.add_axes([0.05, 0.01, 0.85, 0.025])
plt.colorbar(output, orientation='horizontal', cax=position, ticks=range(0, 75, 5))
#plt.colorbar(output, orientation='horizontal', pad=0.025, shrink=0.85)
if saveout:
    plt.tight_layout()
    outfile = os.path.join(outdir, "rad%s_plot.png" % fireyear)
    plt.savefig(outfile,  bbox_inches='tight', dpi=150)


# Out[12]:

# image file:

# Fire candidate pixels

# In[10]:

i_fire, j_fire = np.where(b220.radiance > 5.)
print(zip(i_fire, j_fire))
print(len(i_fire))


# Out[10]:

#     [(37, 164), (37, 165), (37, 169), (38, 164), (38, 165), (64, 91), (88, 156), (88, 157), (116, 216), (116, 217), (117, 216), (117, 217), (118, 216), (118, 217), (135, 252), (135, 253), (136, 252), (136, 253), (144, 233), (144, 234), (145, 232), (145, 233), (145, 234), (146, 226), (146, 227), (146, 232), (146, 233), (147, 225), (147, 226), (147, 227), (147, 233), (148, 224), (148, 225), (148, 226), (148, 232), (148, 233), (149, 223), (149, 224), (149, 227), (149, 228), (150, 223), (150, 227), (150, 228), (150, 229), (150, 231), (150, 232), (151, 223), (151, 227), (151, 228), (151, 229), (151, 231), (151, 232), (151, 233), (152, 228), (152, 229), (152, 230), (152, 231), (152, 232), (152, 233), (153, 228), (153, 229), (153, 230), (153, 231), (153, 232), (153, 233), (154, 227), (154, 228), (154, 229), (154, 230), (157, 230), (158, 230), (159, 181), (160, 180), (160, 181), (160, 182), (160, 183), (160, 194), (161, 181), (161, 182), (161, 183), (161, 194), (161, 195), (161, 204), (161, 205), (162, 182), (162, 183), (162, 204), (162, 205), (162, 206), (163, 204), (163, 205), (163, 206), (163, 207), (164, 204), (164, 205), (164, 206), (164, 212), (165, 204), (165, 205), (165, 206), (165, 212), (167, 187), (168, 187), (168, 205), (168, 206), (169, 205), (169, 206), (170, 202), (170, 205), (170, 206), (171, 141), (171, 142), (171, 202), (171, 203), (171, 205), (171, 206), (171, 207), (171, 208), (172, 141), (172, 142), (172, 202), (172, 203), (172, 204), (172, 208), (173, 141), (173, 142), (173, 143), (173, 202), (173, 203), (173, 204), (173, 205), (174, 204), (174, 205), (175, 204), (175, 205), (176, 175), (177, 175), (179, 179), (180, 178), (180, 179), (180, 180), (181, 178), (181, 179), (183, 93), (183, 184), (184, 105), (184, 184), (184, 185), (185, 105), (185, 184), (185, 185), (185, 186), (186, 182), (186, 183), (186, 185), (187, 82), (187, 181), (187, 182), (187, 183), (188, 181), (188, 182), (188, 183), (188, 184), (189, 180), (189, 181), (189, 182), (189, 183), (189, 184), (190, 180), (190, 181), (191, 180), (191, 181), (192, 107), (192, 108), (192, 186), (192, 187), (193, 107), (193, 108), (193, 109), (193, 185), (193, 186), (193, 187), (194, 107), (194, 108), (194, 109), (194, 186), (194, 187), (195, 45), (195, 48), (195, 64), (195, 91), (195, 92), (195, 108), (196, 48), (196, 49), (196, 64), (196, 88), (196, 89), (196, 91), (196, 92), (196, 93), (197, 49), (197, 50), (197, 51), (197, 64), (197, 65), (197, 91), (197, 92), (197, 93), (198, 49), (198, 50), (198, 64), (198, 65), (198, 66), (198, 92), (199, 65), (200, 124), (201, 85), (201, 86), (201, 88), (201, 89), (201, 124), (201, 125), (202, 85), (202, 86), (202, 89), (202, 163), (202, 164), (202, 285), (203, 89), (203, 165), (204, 89), (204, 165), (204, 166), (205, 164), (205, 165), (205, 185), (205, 186), (206, 185), (206, 186), (206, 187), (207, 185), (207, 186), (207, 187), (208, 186), (208, 187), (210, 185), (210, 186), (211, 85), (211, 137), (212, 85), (212, 86), (212, 87), (212, 137), (213, 85), (213, 86), (213, 87), (213, 88), (213, 89), (213, 137), (214, 86), (214, 87), (214, 88), (214, 89), (214, 137), (215, 87), (215, 88), (215, 91), (215, 136), (215, 180), (215, 182), (216, 90), (216, 91), (216, 92), (216, 93), (216, 179), (216, 180), (216, 181), (216, 182), (216, 183), (217, 91), (217, 92), (217, 93), (217, 94), (217, 180), (217, 181), (218, 91), (218, 92), (218, 93), (218, 158), (218, 159), (219, 93), (219, 158), (219, 159), (220, 159), (225, 94), (226, 94), (226, 95)]
#     298
# 

# Calculate the mask of fire candidate pixels from simple thresholding. Then plot the fire mask.

# In[11]:

firemask = np.zeros(b220.data.shape)
firemask[b220.radiance > 5.] = 1. 


# In[109]:

saveout = False
f, ax = plt.subplots(figsize=(15, 25))
output = ax.pcolor(np.flipud(firemask), cmap='bone', vmin=0, vmax=1)
ax.set_title("Band 220, Fire Candidates with $L > 5 W / m^2 \mu m$")
ax.set_aspect('equal')
ax.set_axis_off()
plt.tight_layout()
if saveout:
    outfile = os.path.join(outdir, "fire01_%s_plot.png" % fireyear)
    plt.savefig(outfile,  bbox_inches='tight', dpi=150)
    plt.close()


# Calculate a square buffer around each pixel in the firemask, N pixels in east/west and north/south direction. (E.g., N=10: 300m). Then plot the buffered mask. 

# In[12]:

N = 10
firemask_ext = np.zeros(b220.data.shape)
for i, j in zip(i_fire, j_fire):
    firemask_ext[i-N:i+N+1, j-N:j+N+1] = 0.5
firemask_ext[b220.radiance > 5.] = 1. 


# In[19]:

saveout = False
f, ax = plt.subplots(figsize=(15, 25))
output = ax.pcolor(np.flipud(firemask_ext), cmap='bone', vmin=0, vmax=1)
ax.set_title("Band 220, Fire Candidates and buffer of %s m" % str(N*30))
ax.set_aspect('equal')
ax.set_axis_off()
plt.tight_layout()
if saveout:
    outfile = os.path.join(outdir, "fire01_buff_%s_plot.png" % fireyear)
    plt.savefig(outfile,  bbox_inches='tight', dpi=150)
    plt.close()


# Out[19]:

# image file:

# Let's save the masks to a TIF file.

# In[116]:

firemask_ds = b220.clone(os.path.join(outdir, '%sfire_01.tif' % firename), firemask_ext) 
firemask_ext_ds = b220.clone(os.path.join(outdir, '%sfire_ext01.tif' % firename), firemask_ext) 


### Fire detection with normalized difference index: HFDI

# Calculate the "classical" HFDI and a variant using band 224, which is calibrated.

# In[13]:

hfdi = hypsc.get_normdiff('band227', 'band191')
hfdi2 = hypsc.get_normdiff('band224', 'band191')


# Out[13]:

#     WARNING:pygaarst.raster:Hyperion band 227 is not calibrated.
#     WARNING:pygaarst.irutils:NaN generated while calculating normalized difference index: : Warning: invalid value encountered in divide
#     
# 

### Fire detection with carbon dioxide absorption feature: CIBR

# Calculate the carbon-dioxide absoption based fire detection index (continuum interpolated band ratio - CIBR).

# In[14]:

cibr = np.divide(hypsc.band185.radiance, 0.666 * hypsc.band183.radiance + 0.334 * hypsc.band189.radiance)
cibr2 = np.divide(hypsc.band185.radiance, 0.666 * hypsc.band183.radiance + 0.334 * hypsc.band188.radiance)


# Out[14]:

#     WARNING:pygaarst.irutils:NaN generated while calculating normalized difference index: : Warning: invalid value encountered in divide
#     
#     WARNING:pygaarst.irutils:NaN generated while calculating normalized difference index: : Warning: invalid value encountered in divide
#     
# 

# In[15]:

print("CIBR min: %.4f, CIBR max: %.4f" % (np.nanmin(cibr), np.nanmax(cibr)))
print("CIBR2 min: %.4f, CIBR2 max: %.4f" % (np.nanmin(cibr2), np.nanmax(cibr2)))


# Out[15]:

#     CIBR min: 0.0022, CIBR max: 49.0000
#     CIBR2 min: 0.0021, CIBR2 max: 49.0000
# 

# Plot CIBR.

# In[16]:

bins = np.linspace(0., 1., num=100)
f, ax = plt.subplots(figsize=(18, 15))
ax.hist(cibr[~np.isnan(cibr)].flatten(), bins=bins)


# Out[16]:

#     (array([  5.33500000e+03,   1.20690000e+04,   2.52500000e+03,
#              2.41000000e+03,   2.55300000e+03,   2.74500000e+03,
#              2.94200000e+03,   3.18000000e+03,   3.31400000e+03,
#              3.36500000e+03,   3.60300000e+03,   3.50900000e+03,
#              3.66100000e+03,   3.74300000e+03,   3.61400000e+03,
#              3.42700000e+03,   3.30600000e+03,   3.23200000e+03,
#              2.91800000e+03,   2.85700000e+03,   2.55400000e+03,
#              2.37800000e+03,   1.99400000e+03,   1.90600000e+03,
#              1.55200000e+03,   1.38800000e+03,   1.23900000e+03,
#              1.06200000e+03,   8.70000000e+02,   7.74000000e+02,
#              5.81000000e+02,   5.64000000e+02,   4.83000000e+02,
#              3.55000000e+02,   3.48000000e+02,   2.77000000e+02,
#              2.30000000e+02,   1.56000000e+02,   1.11000000e+02,
#              7.90000000e+01,   6.70000000e+01,   4.10000000e+01,
#              3.80000000e+01,   3.80000000e+01,   2.40000000e+01,
#              2.40000000e+01,   1.60000000e+01,   1.80000000e+01,
#              1.10000000e+01,   7.00000000e+00,   5.00000000e+00,
#              4.00000000e+00,   3.00000000e+00,   1.00000000e+00,
#              2.00000000e+00,   1.00000000e+00,   0.00000000e+00,
#              1.00000000e+00,   1.00000000e+00,   0.00000000e+00,
#              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
#              1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
#              1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              2.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#              0.00000000e+00,   0.00000000e+00,   6.00000000e+00]),
#      array([ 0.        ,  0.01010101,  0.02020202,  0.03030303,  0.04040404,
#             0.05050505,  0.06060606,  0.07070707,  0.08080808,  0.09090909,
#             0.1010101 ,  0.11111111,  0.12121212,  0.13131313,  0.14141414,
#             0.15151515,  0.16161616,  0.17171717,  0.18181818,  0.19191919,
#             0.2020202 ,  0.21212121,  0.22222222,  0.23232323,  0.24242424,
#             0.25252525,  0.26262626,  0.27272727,  0.28282828,  0.29292929,
#             0.3030303 ,  0.31313131,  0.32323232,  0.33333333,  0.34343434,
#             0.35353535,  0.36363636,  0.37373737,  0.38383838,  0.39393939,
#             0.4040404 ,  0.41414141,  0.42424242,  0.43434343,  0.44444444,
#             0.45454545,  0.46464646,  0.47474747,  0.48484848,  0.49494949,
#             0.50505051,  0.51515152,  0.52525253,  0.53535354,  0.54545455,
#             0.55555556,  0.56565657,  0.57575758,  0.58585859,  0.5959596 ,
#             0.60606061,  0.61616162,  0.62626263,  0.63636364,  0.64646465,
#             0.65656566,  0.66666667,  0.67676768,  0.68686869,  0.6969697 ,
#             0.70707071,  0.71717172,  0.72727273,  0.73737374,  0.74747475,
#             0.75757576,  0.76767677,  0.77777778,  0.78787879,  0.7979798 ,
#             0.80808081,  0.81818182,  0.82828283,  0.83838384,  0.84848485,
#             0.85858586,  0.86868687,  0.87878788,  0.88888889,  0.8989899 ,
#             0.90909091,  0.91919192,  0.92929293,  0.93939394,  0.94949495,
#             0.95959596,  0.96969697,  0.97979798,  0.98989899,  1.        ]),
#      <a list of 99 Patch objects>)

# image file:

# In[142]:

saveout = True
f, ax = plt.subplots(figsize=(15, 25))
output = ax.pcolor(np.flipud(cibr), cmap='hot', vmin=0, vmax=5)
ax.set_title("%s %s fire: CIBR" % (fireyear, firename.capitalize()))
ax.set_aspect('equal')
ax.set_axis_off()
position=f.add_axes([0.05, 0.2, 0.85, 0.025])
#position=f.add_axes([0.05, 0.01, 0.85, 0.025])
plt.colorbar(output, orientation='horizontal', cax=position, ticks=range(0, 75, 5))
plt.tight_layout()
if saveout:
    outfile = os.path.join(outdir, "cibr_%s_plot.png" % fireyear)
    plt.savefig(outfile,  bbox_inches='tight', dpi=150)
    plt.close()


# Out[142]:

#     WARNING:pygaarst.irutils:NaN generated while calculating normalized difference index: : Warning: invalid value encountered in less
#     
# 

### Fire detection with potassium emission feature: Kemiss

# In[17]:

kemiss = np.divide(hypsc.band42.radiance, hypsc.band43.radiance)


# Out[17]:

#     WARNING:pygaarst.irutils:NaN generated while calculating normalized difference index: : Warning: invalid value encountered in divide
#     
# 

### Calculate spectra

#### Preparation

# How long does it take to calculate 1 spectrum? This is mostly an I/O problem as the individual bands still reside in different data files.

# In[18]:

get_ipython().run_cell_magic(u'timeit', u'', u"spec1 = hypsc.spectrum(i_fire[45], j_fire[45], bands='selected', bdsel=range(127, 224))")


# Out[18]:

#     1 loops, best of 3: 268 ms per loop
# 

#### Fire pixels (candidates)

# Calculate default spectra for fire pixels. Default means only calibrated bands. How many did we get?

# In[117]:

with open(infn, "rU") as source:
    specs = pickle.load(source)


# In[18]:

loadfromfile = True
calculate = False
if loadfromfile:
    try:
        infn = os.path.join(outdir, "%s_firespecs.dat" % firename)
        with open(infn, "rU") as source:
            specs = pickle.load(source)
    except IOError:
        calculate = True
if (calculate or not loadfromfile):
    specs = [hypsc.spectrum(i, j) for i, j in zip(i_fire, j_fire)]


# In[19]:

print(len(specs), len(i_fire))
print(type(specs[0][0]), len(specs[0]))
print(type(specs), type(specs[0]))


# Out[19]:

#     (298, 298)
#     (<type 'numpy.float32'>, 198)
#     (<type 'list'>, <type 'list'>)
# 

# Alternative: save spectra to HDF5 file.

# In[37]:

outfn = os.path.join(outdir, "%s_firespecs_calib.hdf5" % firename)
save_to_hdf5(outfn, hypsc, specs, i_fire, j_fire)


# In[32]:

np.where(hypsc.band_is_calibrated)


# Out[32]:

#     (array([  7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
#             20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
#             33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
#             46,  47,  48,  50,  52,  54,  56,  58,  60,  61,  62,  63,  64,
#             65,  67,  69,  71,  73,  75,  77,  79,  81,  83,  85,  87,  89,
#             91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
#            104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
#            117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
#            130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
#            143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
#            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
#            169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
#            182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
#            195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
#            208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
#            221, 222, 223]),)

# Pickle results.

# In[51]:

import pickle
out = os.path.join(outdir, "%s_firespecs.dat" % firename)
with open(out, "w") as sink:
    pickle.dump(specs, sink)


# Calculate high wavelength (uncalibrated), and test. 

# In[ ]:

specs_high = [hypsc.spectrum(i, j, 'high') for i, j in zip(i_fire, j_fire)]


# In[22]:

print(len(specs_high), len(i_fire))


# Out[22]:

#     (298, 298)
# 

# In[29]:

highwavelengths = hypsc.hyperionwavelength_nm[225:]
print(highwavelengths)
print(len(highwavelengths), len(specs_high[2]))
print(specs_high[2])


# Out[29]:

#     [ 2415.7   2425.8   2435.89  2445.99  2456.09  2466.09  2476.19  2486.29
#       2496.39  2506.48  2516.59  2526.68  2536.78  2546.88  2556.98  2566.98
#       2577.08]
#     (17, 17)
#     [0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125]
# 

# As it turns out, the high uncalibrated spectra don't contain any usable data at all. We'll neglect them from now on. 

#### Buffer pixels

# Pull sample from buffer pixels (that is, not fire pixels, within buffer from fire pixels) if there is no saved data file. These represent "background" candidates. 

# In[87]:

N=100
loadfromfile = True
if loadfromfile:
    fn = os.path.join(outdir, "%s_buffersample.txt" % firename)
    with open(fn, "rU") as source:
        for line in source:
            if line.startswith('#'):
                continue
            buffersample = json.loads(line)
else:
    i_buffer, j_buffer = np.where(firemask_ext == 0.5)
    buffersample = random.sample(zip(i_buffer, j_buffer), N)
print(buffersample)


# Out[87]:

#     [[200, 62], [218, 105], [156, 200], [194, 275], [188, 44], [201, 104], [177, 142], [70, 88], [202, 179], [33, 170], [232, 105], [189, 72], [183, 205], [160, 206], [203, 184], [152, 191], [199, 84], [189, 46], [195, 84], [121, 218], [217, 158], [97, 166], [172, 133], [203, 108], [164, 198], [184, 208], [211, 194], [108, 225], [163, 182], [155, 200], [66, 86], [143, 263], [112, 210], [187, 90], [200, 89], [137, 219], [193, 113], [134, 231], [185, 170], [125, 212], [222, 96], [196, 120], [219, 85], [212, 183], [174, 84], [210, 176], [180, 97], [184, 77], [162, 137], [182, 79], [228, 103], [175, 87], [211, 280], [167, 133], [153, 209], [82, 155], [169, 149], [127, 212], [141, 251], [187, 110], [195, 109], [224, 94], [185, 192], [194, 94], [214, 139], [205, 290], [203, 70], [36, 165], [96, 150], [196, 82], [48, 169], [167, 204], [199, 95], [139, 244], [110, 216], [218, 83], [167, 152], [186, 113], [175, 137], [190, 67], [172, 210], [187, 85], [128, 218], [221, 189], [206, 163], [174, 177], [169, 196], [178, 172], [206, 123], [218, 146], [134, 226], [32, 167], [116, 208], [178, 153], [179, 193], [206, 282], [208, 69], [122, 225], [165, 144], [196, 98]]
# 

# Save buffer sample to file.

# In[119]:

import json
out = os.path.join(outdir, "%s_buffersample.txt" % firename)
with open(out, "w") as sink:
    sink.write("#i and j coordinates for random sample from buffer pixels around fire candidate pixels, %s\n" % firename.capitalize())
    sink.write(json.dumps(buffersample))


# Calculate spectra at buffer sample pixels.

# In[88]:

loadfromfile = True

if loadfromfile:
    try:
        infn = os.path.join(outdir, "%s_bufferspecs.dat" % firename)
        with open(infn, "rU") as source:
            specs_buffer = pickle.load(source)
    except IOError:
        calculate = True
if (calculate or not loadfromfile):
    specs_buffer = [hypsc.spectrum(i, j) for i, j in buffersample]


# Alternatively, calculate spectra for ALL buffer pixels. 

# In[48]:

i_buffer, j_buffer = np.where(firemask_ext == 0.5)
bufferall = zip(i_buffer, j_buffer)
print(len(bufferall))

loadfromfile = False

if loadfromfile:
    try:
        infn = os.path.join(outdir, "%s_bufferallspecs.dat" % firename)
        with open(infn, "rU") as source:
            specs_bufferall = pickle.load(source)
    except IOError:
        calculate = True
if (calculate or not loadfromfile):
    specs_bufferall = [hypsc.spectrum(i, j) for i, j in bufferall]


# Out[48]:

#     11962
# 

# Pickle spectra.

# In[ ]:

out = os.path.join(outdir, "%s_bufferspecs.dat" % firename)
with open(out, "w") as sink:
    pickle.dump(specs_buffer, sink)


# In[19]:

out = os.path.join(outdir, "%s_bufferallspecs_calib.dat" % firename)
with open(out, "w") as sink:
    pickle.dump(bufferall, sink)


# Save to HDF5 file.

# In[50]:

outfn = os.path.join(outdir, "%s_bufferallspecs_calib.hdf5" % firename)
save_to_hdf5(outfn, hypsc, specs_bufferall, i_buffer, j_buffer)


#### Outside pixels

# In[128]:

N=100
loadfromfile = True
if loadfromfile:
    fn = os.path.join(outdir, "%s_outsidesample.txt" % firename)
    with open(fn, "rU") as source:
        for line in source:
            if line.startswith('#'):
                continue
            outsidesample = json.loads(line)
else:
    i_outside, j_outside = np.where(firemask_ext == 0)
    outsidesample = random.sample(zip(i_outside, j_outside), N)
print(outsidesample)


# Out[128]:

#     [[133, 113], [304, 305], [272, 260], [110, 314], [217, 313], [115, 167], [161, 317], [60, 271], [101, 294], [129, 301], [16, 274], [33, 252], [129, 96], [157, 321], [340, 261], [145, 344], [255, 267], [135, 339], [279, 207], [4, 42], [332, 291], [15, 184], [70, 49], [86, 16], [168, 326], [150, 22], [8, 87], [305, 208], [268, 26], [45, 211], [153, 8], [27, 189], [94, 41], [120, 294], [207, 17], [95, 101], [83, 10], [335, 82], [344, 200], [195, 245], [214, 293], [242, 49], [239, 254], [135, 338], [233, 353], [147, 274], [275, 39], [197, 211], [290, 81], [127, 21], [284, 295], [262, 71], [25, 210], [114, 128], [145, 5], [327, 302], [58, 353], [162, 44], [94, 354], [232, 366], [313, 136], [221, 355], [278, 24], [18, 273], [23, 26], [70, 154], [161, 272], [40, 255], [83, 7], [218, 257], [220, 215], [25, 300], [57, 223], [165, 262], [0, 178], [59, 161], [114, 282], [333, 72], [64, 129], [323, 345], [343, 132], [190, 306], [185, 17], [110, 323], [244, 208], [293, 182], [241, 265], [118, 304], [245, 2], [203, 349], [293, 318], [184, 237], [344, 177], [301, 256], [338, 66], [97, 130], [88, 341], [128, 62], [298, 311], [75, 247]]
# 

# In[127]:

import json
out = os.path.join(outdir, "%s_outsidesample.txt" % firename)
with open(out, "w") as sink:
    sink.write("#i and j coordinates for random sample from pixels other than fire or buffer pixel, %s\n" % firename.capitalize())
    sink.write(json.dumps(outsidesample))


# In[129]:

loadfromfile = True

if loadfromfile:
    try:
        infn = os.path.join(outdir, "%s_outsidespecs.dat" % firename)
        with open(infn, "rU") as source:
            specs_outside = pickle.load(source)
    except IOError:
        calculate = True
if (calculate or not loadfromfile):
    specs_outside = [hypsc.spectrum(i, j) for i, j in outsidesample]


# Pickle spectra.

# In[130]:

out = os.path.join(outdir, "%s_outsidespecs.dat" % firename)
with open(out, "w") as sink:
    pickle.dump(specs_outside, sink)


### Plot spectra

# Fire candidate pixels first: loop through fire (candidate) pixel sectra, retrieve mod HDFI and CIBR, plot. 

# In[64]:

saveout = False
for idx, spec in enumerate(specs[50:55]):
    f, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral radiance")
    ax.set_xlim(400, 2500)
    ax.set_ylim(0, 70)
    ax.set_title("%s %s fire: Spectrum for fire candidate pixel at %s, %s" % (fireyear, firename.capitalize(), i_fire[idx], j_fire[idx]))
    ax.plot(hypsc.calibratedwavelength_nm, spec)  
    #ax.plot(highwavelengths, specs_high[idx], linewidth=5)  
    ax.vlines([770, 2010, 2060, 2430], [0, 0, 0, 0], [70, 70, 70, 70])
    #ax.text(1600, 60, "HFDI-value: %.4f" % hfdi[i_fire[idx], j_fire[idx]])
    ax.text(1600, 54, "mod HFDI-value: %.4f" % hfdi2[i_fire[idx], j_fire[idx]])
    ax.text(1600, 60, "CIBR-value: %.4f" % cibr[i_fire[idx], j_fire[idx]])
    ax.text(1600, 57, "mod CIBR-value: %.4f" % cibr2[i_fire[idx], j_fire[idx]])
    if saveout:
        outfile = os.path.join(outdir, "%s_spectrum_full_%s_%s.png" % (firename, i_fire[idx], j_fire[idx]))
        plt.savefig(outfile,  bbox_inches='tight', dpi=150)
        plt.close()


# Out[64]:

# image file:

# image file:

# image file:

# image file:

# image file:

# Plot spectra around the CO2 absorption feature to investigate it further.

# In[131]:

idx_co2, bandname, hyp_co2 = find_nearest_hyp(2010)
idx_spec = np.where(hypsc.calibratedwavelength_nm == hyp_co2)[0]
saveout = False
f, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral radiance")
ax.set_xlim(1930, 2090)
ax.set_ylim(0, 50)
ax.set_title("%s %s fire: Carbon dioxide feature location with a sample of fire pixel spectra" % (fireyear, firename.capitalize()))
ax.vlines(2010, 0, 70)
ax.annotate('band 185', xy=(2002, 35), xytext=(2002, 45),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
ax.annotate('band 188', xy=(2032, 35), xytext=(2032, 45),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
ax.annotate('band 183', xy=(1982, 35), xytext=(1982, 45),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
for idx, spec in enumerate(specs[100:150]):
    ax.plot(hypsc.calibratedwavelength_nm, spec, 's-')  
    ax.scatter(hyp_co2, spec[idx_spec], zorder=5, marker='o', s=100)
#    ax.plot(highwavelengths, specs_high[idx], linewidth=5)  
    if saveout:
        outfile = os.path.join(outdir, "%s_co2_fire_sample.png" % firename)
        plt.savefig(outfile,  dpi=150)
        plt.close()


# Out[131]:

# image file:

# Plot spectra around the potassium emission feature around 770 nm.

# In[138]:

idx_k, bandname_k, hyp_k = find_nearest_hyp(770)
idx_spec = np.where(hypsc.calibratedwavelength_nm == hyp_k)[0]
saveout = False
f, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral radiance")
ax.set_xlim(720, 820)
ax.set_ylim(0, 70)
ax.set_title("%s %s fire: Potassium feature location with a sample of fire pixel spectra" % (fireyear, firename.capitalize()))
ax.vlines(770, 0, 70)
#ax.annotate('band 185', xy=(2002, 35), xytext=(2002, 45),
#        arrowprops=dict(facecolor='black', shrink=0.05),
#        )
#ax.annotate('band 188', xy=(2032, 35), xytext=(2032, 45),
#        arrowprops=dict(facecolor='black', shrink=0.05),
#        )
#ax.annotate('band 183', xy=(1982, 35), xytext=(1982, 45),
#        arrowprops=dict(facecolor='black', shrink=0.05),
#        )
for idx, spec in enumerate(specs[100:150]):
    ax.plot(hypsc.calibratedwavelength_nm, spec, 's-')  
    ax.scatter(hyp_co2, spec[idx_spec], zorder=5, marker='o', s=100)
#    ax.plot(highwavelengths, specs_high[idx], linewidth=5)  
    if saveout:
        outfile = os.path.join(outdir, "%s_K_fire_sample.png" % firename)
        plt.savefig(outfile,  dpi=150)
        plt.close()


# Out[138]:

# image file:

# Plot spectra for buffer pixels. 

# In[95]:

saveout = True
for idx, spec in enumerate(specs_buffer):
    f, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral radiance")
    ax.set_xlim(400, 2500)
    ax.set_ylim(0, 70)
    ax.set_title("%s %s fire: Spectrum for background candidate pixel at %s, %s" % (fireyear, firename.capitalize(), buffersample[idx][0], buffersample[idx][1]))
    ax.plot(hypsc.calibratedwavelength_nm, spec)  
    ax.vlines([770, 2010, 2060, 2430], [0, 0, 0, 0], [70, 70, 70, 70])
    ax.text(1600, 54, "mod HFDI-value: %.4f" % hfdi2[i_fire[idx], j_fire[idx]])
    ax.text(1600, 60, "CIBR-value: %.4f" % cibr[i_fire[idx], j_fire[idx]])
    ax.text(1600, 57, "mod CIBR-value: %.4f" % cibr2[i_fire[idx], j_fire[idx]])
    if saveout:
        outfile = os.path.join(outdir, "%s_spectrum_full_buffer_%s_%s.png" % (firename, i_fire[idx], j_fire[idx]))
        plt.savefig(outfile,  bbox_inches='tight', dpi=150)
        plt.close()


# In[35]:

idx_co2, bandname, hyp_co2 = find_nearest_hyp(2010)
idx_spec = np.where(hypsc.calibratedwavelength_nm == hyp_co2)[0]
saveout = False
f, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral radiance")
ax.set_xlim(1930, 2090)
ax.set_ylim(0, 10)
ax.set_title("%s %s fire: Carbon dioxide feature location with a sample of background spectra" % (fireyear, firename.capitalize()))
ax.vlines(2010, 0, 10)
ax.annotate('band 185', xy=(2002, 5), xytext=(2002, 7),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
ax.annotate('band 188', xy=(2032, 5), xytext=(2032, 7),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
ax.annotate('band 183', xy=(1982, 5), xytext=(1982, 7),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
for idx, spec in enumerate(specs_buffer[:10]):
    ax.plot(hypsc.calibratedwavelength_nm, spec, 's-')  
    ax.scatter(hyp_co2, spec[idx_spec], zorder=5, marker='o', s=100)
    if saveout:
        outfile = os.path.join(outdir, "%s_spectrum_full_%s_%s.png" % (firename, i_fire[idx], j_fire[idx]))
        plt.savefig(outfile,  bbox_inches='tight', dpi=150)
        plt.close()


# Out[35]:

# image file:

# Now for outside pixels.

# In[143]:

idx_co2, bandname, hyp_co2 = find_nearest_hyp(2010)
idx_spec = np.where(hypsc.calibratedwavelength_nm == hyp_co2)[0]
saveout = False
f, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral radiance")
ax.set_xlim(1930, 2090)
ax.set_ylim(0, 10)
ax.set_title("%s %s fire: Carbon dioxide feature location with a sample of spectra far away from fire" % (fireyear, firename.capitalize()))
ax.vlines(2010, 0, 10)
ax.annotate('band 185', xy=(2002, 5), xytext=(2002, 7),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
ax.annotate('band 188', xy=(2032, 5), xytext=(2032, 7),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
ax.annotate('band 183', xy=(1982, 5), xytext=(1982, 7),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
for idx, spec in enumerate(specs_outside[:10]):
    ax.plot(hypsc.calibratedwavelength_nm, spec, 's-')  
    ax.scatter(hyp_co2, spec[idx_spec], zorder=5, marker='o', s=100)
    if saveout:
        outfile = os.path.join(outdir, "%s_spectrum_outside_full_%s_%s.png" % (firename, i_fire[idx], j_fire[idx]))
        plt.savefig(outfile,  bbox_inches='tight', dpi=150)
        plt.close()


# Out[143]:

# image file:

# In[137]:

idx_k, bandname_k, hyp_k = find_nearest_hyp(770)
idx_spec = np.where(hypsc.calibratedwavelength_nm == hyp_k)[0]
saveout = False
f, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral radiance")
ax.set_xlim(720, 820)
ax.set_ylim(0, 80)
ax.set_title("%s %s fire: Potassium feature location with a sample of spectra far away from fire" % (fireyear, firename.capitalize()))
ax.vlines(770, 0, 70)
ax.annotate(bandname_k, xy=(hyp_k, 65), xytext=(hyp_k, 75),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
#ax.annotate('band 188', xy=(2032, 35), xytext=(2032, 45),
#        arrowprops=dict(facecolor='black', shrink=0.05),
#        )
#ax.annotate('band 183', xy=(1982, 35), xytext=(1982, 45),
#        arrowprops=dict(facecolor='black', shrink=0.05),
#        )
for idx, spec in enumerate(specs_outside[10:60]):
    ax.plot(hypsc.calibratedwavelength_nm, spec, 's-')  
    ax.scatter(hyp_co2, spec[idx_spec], zorder=5, marker='o', s=100)
#    ax.plot(highwavelengths, specs_high[idx], linewidth=5)  
    if saveout:
        outfile = os.path.join(outdir, "%s_K_outside_sample.png" % firename)
        plt.savefig(outfile,  dpi=150)
        plt.close()


# Out[137]:

# image file:

### Quality of fire indices

# Average CIBR for fire and buffer pixels.

# In[132]:

cibrs_outside = np.array([cibr2[i, j] for i, j in outsidesample])
cibrs_buff = np.array([cibr2[i, j] for i, j in buffersample])
cibrs_fire = np.array([cibr2[i, j] for i, j in zip(i_fire, j_fire)])
mean_outside, std_outside = np.mean(cibrs_outside[~np.isnan(cibrs_outside)]), np.std(cibrs_outside[~np.isnan(cibrs_outside)])
mean_buff, std_buff = np.mean(cibrs_buff[~np.isnan(cibrs_buff)]), np.std(cibrs_buff[~np.isnan(cibrs_buff)])
mean_fire, std_fire = np.mean(cibrs_fire), np.std(cibrs_fire)
print(mean_outside, std_outside)
print(mean_buff, std_buff)
print(mean_fire, std_fire)


# Out[132]:

#     (0.11061936, 0.082325585)
#     (0.12504987, 0.083247416)
#     (0.33778536, 0.079915613)
# 

# In[134]:

f, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.plot(np.linspace(-1, 1, 100), stat.norm.pdf(np.linspace(-1, 1, 100), mean_buff, std_buff), label='buffer zone')
ax.plot(np.linspace(-1, 1, 100), stat.norm.pdf(np.linspace(-1, 1, 100), mean_fire, std_fire), label='fire')
ax.plot(np.linspace(-1, 1, 100), stat.norm.pdf(np.linspace(-1, 1, 100), mean_outside, std_outside), label='outside buffer zone')
ax.legend()
ax.set_title("Modified $CO_2$ absorption CIBR index, %s fire" % firename.capitalize())


# Out[134]:

#     <matplotlib.text.Text at 0x129bcb410>

# image file:

# Same for mod HDMI

# In[135]:

hfdi2s_outside = np.array([hfdi2[i, j] for i, j in outsidesample])
hfdi2s_buff = np.array([hfdi2[i, j] for i, j in buffersample])
hfdi2s_fire = np.array([hfdi2[i, j] for i, j in zip(i_fire, j_fire)])
mean_outside, std_outside = np.mean(hfdi2s_outside[~np.isnan(hfdi2s_outside)]), np.std(hfdi2s_outside[~np.isnan(hfdi2s_outside)])
mean_buff, std_buff = np.mean(hfdi2s_buff[~np.isnan(hfdi2s_buff)]), np.std(hfdi2s_buff[~np.isnan(hfdi2s_buff)])
mean_fire, std_fire = np.mean(hfdi2s_fire), np.std(hfdi2s_fire)
print(mean_outside, std_outside)
print(mean_buff, std_buff)
print(mean_fire, std_fire)


# Out[135]:

#     (-0.31852448, 0.36618954)
#     (-0.28128362, 0.32938206)
#     (0.053092424, 0.1392234)
# 

# In[136]:

f, ax = plt.subplots(1, 1, figsize=(15, 12))
ax.plot(np.linspace(-1, 1, 100), stat.norm.pdf(np.linspace(-1, 1, 100), mean_buff, std_buff), label='buffer zone')
ax.plot(np.linspace(-1, 1, 100), stat.norm.pdf(np.linspace(-1, 1, 100), mean_fire, std_fire), label='fire')
ax.plot(np.linspace(-1, 1, 100), stat.norm.pdf(np.linspace(-1, 1, 100), mean_outside, std_outside), label='outside buffer zone')
ax.legend()
ax.set_title("Modified HFDI, %s fire" % firename.capitalize())


# Out[136]:

#     <matplotlib.text.Text at 0x108df6d10>

# image file:

# Same for kemiss

# In[139]:

kemiss_outside = np.array([kemiss[i, j] for i, j in outsidesample])
kemiss_buff = np.array([kemiss[i, j] for i, j in buffersample])
kemiss_fire = np.array([kemiss[i, j] for i, j in zip(i_fire, j_fire)])
mean_outside, std_outside = np.mean(kemiss_outside[~np.isnan(kemiss_outside)]), np.std(kemiss_outside[~np.isnan(kemiss_outside)])
mean_buff, std_buff = np.mean(kemiss_buff[~np.isnan(kemiss_buff)]), np.std(kemiss_buff[~np.isnan(kemiss_buff)])
mean_fire, std_fire = np.mean(kemiss_fire), np.std(kemiss_fire)
print(mean_outside, std_outside)
print(mean_buff, std_buff)
print(mean_fire, std_fire)


# Out[139]:

#     (0.97515625, 0.029731005)
#     (0.9930675, 0.020570677)
#     (0.99585849, 0.016047871)
# 

# In[140]:

f, ax = plt.subplots(1, 1, figsize=(15, 12))
ax.plot(np.linspace(0.7, 1.3, 100), stat.norm.pdf(np.linspace(0.7, 1.7, 100), mean_buff, std_buff), label='buffer zone')
ax.plot(np.linspace(0.7, 1.3, 100), stat.norm.pdf(np.linspace(0.7, 1.7, 100), mean_fire, std_fire), label='fire')
ax.plot(np.linspace(0.7, 1.3, 100), stat.norm.pdf(np.linspace(0.7, 1.7, 100), mean_outside, std_outside), label='outside buffer zone')
ax.legend()
ax.set_title("Potassium emission index, %s fire" % firename.capitalize())


# Out[140]:

#     <matplotlib.text.Text at 0x10ed1f9d0>

# image file:

### Classify spectra

# In[85]:

from sklearn.cluster import DBSCAN
from sklearn import metrics

specfirearray = np.array(specs)
specbufferarray = np.array(specs_buffer)
specdata = np.concatenate((specfirearray[:,125:], specbufferarray[:,125:]), axis=0)
db = DBSCAN(min_samples=4).fit(specdata)
core_samples = db.core_sample_indices_
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)


# Out[85]:


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-85-9aa722964bee> in <module>()
          3 
          4 specfirearray = np.array(specs)
    ----> 5 specbufferarray = np.array(specs_buffer)
          6 specdata = np.concatenate((specfirearray[:,125:], specbufferarray[:,125:]), axis=0)
          7 db = DBSCAN(min_samples=4).fit(specdata)


    NameError: name 'specs_buffer' is not defined


# In[84]:

print(core_samples)
print(labels)


# Out[84]:

#     [9548, 10585, 10584, 8676, 8849, 9021, 9022, 9198, 9199, 9374, 9375, 9549, 9550, 9711, 9712, 9713, 9873, 9874, 9875, 9876, 10042, 10043, 10044, 10045, 10197, 10198, 10199, 10200, 10329, 10330, 10331, 10332, 10333, 10462, 10463, 10464, 10465, 10466, 10581, 10582, 10583]
#     [-1. -1. -1. ..., -1. -1. -1.]
# 

### The built-in Hyperion band data

# In[14]:

hypdata = hyp.gethyperionbands()
print(hypdata.dtype.names)


# Out[14]:

#     ('Hyperion_Band', 'Average_Wavelength_nm', 'Full_Width_at_Half_the_Maximum_FWHM_nm', 'Spatial_Resolution_m', 'Not_Calibrated_X')
# 

# In[25]:

np.logical_not(hypdata.Not_Calibrated_X == 'X')


# Out[25]:

#     array([False, False, False, False, False, False, False,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True, False,  True, False,  True, False,
#             True, False,  True, False,  True, False,  True,  True,  True,
#             True,  True,  True, False,  True, False,  True, False,  True,
#            False,  True, False,  True, False,  True, False,  True, False,
#             True, False,  True, False,  True, False,  True, False,  True,
#            False,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True,  True,
#             True,  True,  True,  True,  True,  True,  True,  True, False,
#            False, False, False, False, False, False, False, False, False,
#            False, False, False, False, False, False, False, False], dtype=bool)

# In[26]:

print(hypdata.Average_Wavelength_nm )


# Out[26]:

#     [  355.59   365.76   375.94   386.11   396.29   406.46   416.64   426.82
#        436.99   447.17   457.34   467.52   477.69   487.87   498.04   508.22
#        518.39   528.57   538.74   548.92   559.09   569.27   579.45   589.62
#        599.8    609.97   620.15   630.32   640.5    650.67   660.85   671.02
#        681.2    691.37   701.55   711.72   721.9    732.07   742.25   752.43
#        762.6    772.78   782.95   793.13   803.3    813.48   823.65   833.83
#        844.     851.92   854.18   862.01   864.35   872.1    874.53   882.19
#        884.7    892.28   894.88   902.36   905.05   912.45   915.23   922.54
#        925.41   932.64   935.58   942.73   945.76   952.82   955.93   962.91
#        966.11   972.99   976.28   983.08   986.46   993.17   996.63  1003.3
#       1006.81  1013.3   1016.98  1023.4   1027.16  1033.49  1037.33  1043.59
#       1047.51  1053.69  1057.68  1063.79  1073.89  1083.99  1094.09  1104.19
#       1114.19  1124.28  1134.38  1144.48  1154.58  1164.68  1174.77  1184.87
#       1194.97  1205.07  1215.17  1225.17  1235.27  1245.36  1255.46  1265.56
#       1275.66  1285.76  1295.86  1305.96  1316.05  1326.05  1336.15  1346.25
#       1356.35  1366.45  1376.55  1386.65  1396.74  1406.84  1416.94  1426.94
#       1437.04  1447.14  1457.23  1467.33  1477.43  1487.53  1497.63  1507.73
#       1517.83  1527.92  1537.92  1548.02  1558.12  1568.22  1578.32  1588.42
#       1598.51  1608.61  1618.71  1628.81  1638.81  1648.9   1659.    1669.1
#       1679.2   1689.3   1699.4   1709.5   1719.6   1729.7   1739.7   1749.79
#       1759.89  1769.99  1780.09  1790.19  1800.29  1810.38  1820.48  1830.58
#       1840.58  1850.68  1860.78  1870.87  1880.98  1891.07  1901.17  1911.27
#       1921.37  1931.47  1941.57  1951.57  1961.66  1971.76  1981.86  1991.96
#       2002.06  2012.15  2022.25  2032.35  2042.45  2052.45  2062.55  2072.65
#       2082.75  2092.84  2102.94  2113.04  2123.14  2133.24  2143.34  2153.34
#       2163.43  2173.53  2183.63  2193.73  2203.83  2213.93  2224.03  2234.12
#       2244.22  2254.22  2264.32  2274.42  2284.52  2294.61  2304.71  2314.81
#       2324.91  2335.01  2345.11  2355.21  2365.2   2375.3   2385.4   2395.5
#       2405.6   2415.7   2425.8   2435.89  2445.99  2456.09  2466.09  2476.19
#       2486.29  2496.39  2506.48  2516.59  2526.68  2536.78  2546.88  2556.98
#       2566.98  2577.08]
# 

# In[29]:

find_nearest_hyp(800)


# Out[29]:

#     (44, 'band45', 803.29999999999995)

# In[30]:

find_nearest_hyp(2010)


# Out[30]:

#     (185, 'band186', 2012.1500000000001)

# In[ ]:

boundaryrad220 = a.clone(os.path.join(outdir, 'boundaryrad220.tif'), a.radiance) 


### MODTRAN plotting

# In[42]:

plotfile = "/Volumes/WORKING/Modtran/tape7"


# In[43]:

wavenum, transmiss = np.genfromtxt(plotfile, unpack=True, skiprows=12, usecols=(0, 1), skip_footer=1)


# In[44]:

wavenum


# Out[44]:

#     array([  3448.,   3449.,   3450., ...,  19998.,  19999.,  20000.])

# In[45]:

transmiss


# Out[45]:

#     array([ 0.    ,  0.0029,  0.0321, ...,  0.2399,  0.2398,  0.2398])

# In[46]:

wav_mod = 1./wavenum * 1.e4


# In[47]:

wav_mod


# Out[47]:

#     array([ 2.90023202,  2.89939113,  2.89855072, ...,  0.50005001,
#             0.500025  ,  0.5       ])

# In[48]:

f, ax = plt.subplots(figsize=(15, 10))
ax.plot(wav_mod[::-1], transmiss[::-1])
ax.set_title("MODTRAN4 run output for sample atmosphere: sub-arctic summer, 5 km visibility")
ax.set_xlabel("Wavelength in $\mu m$")
ax.set_ylabel("Transmissivity (between 0 and 1)")
outfile = os.path.join(outdir, "MODTRANout_01.png") 
plt.savefig(outfile, bbox_inches='tight', dpi=150)


# Out[48]:

# image file:

# In[49]:

nu = 3448.
1/nu * 1e8


# Out[49]:

#     29002.320185614848

# In[79]:

T = 700 # fire temperature, K
hypdata = hyp.gethyperionbands()
hypdata = hypdata[np.logical_not(hypdata.Not_Calibrated_X == 'X')]
lambdabands_hyp = hypdata.Average_Wavelength_nm
fwhms_hyp = hypdata.Full_Width_at_Half_the_Maximum_FWHM_nm
lambdas = []
atmoplancks = []
for lambdaband, fwhm in zip(lambdabands_hyp, fwhms_hyp):
    inbetween = np.logical_and(wav_mod*1000 > lambdaband - fwhm/2., wav_mod*1000 < lambdaband + fwhm/2.)
    lambdamod = wav_mod[inbetween]
    transmissmod = transmiss[inbetween]
    N = len(lambdamod)
    if not lambdamod.size:
        continue
    dummy = np.sum(specrad(lambdamod, T) * transmissmod)/N
    lambdas.append(lambdaband)
    atmoplancks.append(dummy)
    
plancks = np.array([specrad(l/1000, T) for l in lambdas])
f, ax = plt.subplots(figsize=(15, 10))
ax.plot(lambdas, atmoplancks)
ax.plot(lambdas, plancks)
ax.set_xlim(500, 2800)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral radiance ($W / (\mu m \ m^2\,sr)$)")
ax.set_title("Blackbody and atmospherically corrected emission spectrum for T=%s K" % T)
    
    
        


# Out[79]:

#     <matplotlib.text.Text at 0x109f39d50>

# image file:

# In[13]:

hypdata.dtype.names


# Out[13]:

#     ('Hyperion_Band',
#      'Average_Wavelength_nm',
#      'Full_Width_at_Half_the_Maximum_FWHM_nm',
#      'Spatial_Resolution_m',
#      'Not_Calibrated_X')

### Other: Quick-and-dirty landcover classification with LTK algorithm

# This should be re-done with Hyperion. For the moment, just to look at it, I'm using the Landsat 7 scenes. 

# In[25]:

if fireyear == 2004:
    landsatsc = raster.Landsatscene(os.path.join(basedir, dir_2004, landsatscene_2004))
elif fireyear == 2009:
    landsatsc = raster.Landsatscene(os.path.join(basedir, dir_2009, landsatscene_2009))

landsatsc.infix = '_CLIP'

saveout = True
cmap = mpl.colors.ListedColormap(['black', 'tan', 'red', 'deepskyblue', 'whitesmoke', 'olivedrab'])
bounds = range(7)
cmap.set_under('black')
cmap.set_over('olivedrab')
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cmap1 = mpl.colors.ListedColormap(['olivedrab', 'whitesmoke', 'black'])
bounds1 = range(4)
norm1 = mpl.colors.BoundaryNorm(bounds1, cmap1.N)
labels = np.arange(0, 7, 1)
labeltext = ['no data', 'bare soil', 'ice/snow', 'water', 'cloud', 'vegetation']
loc = 0.8*labels + .5

stripemask = (landsatsc.band4.data == 0)
cloudclassification = landsatsc.ltkcloud
cloudclassification[stripemask] = 0.
fig = plt.figure(figsize=(18, 18))
plt.imshow(cloudclassification, cmap=cmap, norm=norm)
plt.title("%s LTK land cover classification" % landsatsc.dirname)
plt.imshow(cloudclassification, cmap=cmap)
cb = plt.colorbar(orientation='horizontal', shrink=0.9, pad=0.05)
cb.set_ticks(loc)
cb.set_ticklabels(labeltext)

if saveout:
    outfile = "%s_cloud_ltk" % firename
    plt.savefig(outfile, bbox_inches='tight', dpi=150)


# Out[25]:

#     WARNING:root:Metadata file /Volumes/SCIENCE/Fire/DATA_BY_PROJECT/HyspIRI_Prep_2014/2004Boundary/LE70690142004201EDC01/LE70690142004201EDC01_MTL.txt appears to have extra lines after the end of the metadata. This is probably, but not necessarily, harmless.
# 

# image file:

### Sandbox

# In[111]:

fig = plt.figure(figsize=(15, 15))
f, ax = plt.subplots()
dat = np.ma.masked_where(a.radiance < 10, a.radiance) 
freq, bins, patches = ax.hist(dat.compressed(), bins=1000)


# Out[111]:

# image file:

# image file:

# In[109]:

ma.masked_outside(a.radiance, .1, 500).mean()


# Out[109]:

#     1.7436380807549834

# In[12]:

np.divide(np.nan, np.nan)


# Out[12]:

#     nan

# In[84]:

np.zeros(5,dtype=np.float32)


# Out[84]:

#     array([ 0.,  0.,  0.,  0.,  0.], dtype=float32)

# In[85]:

np.divide(0., 0.)


# Out[85]:


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-85-a3a7936f7e92> in <module>()
    ----> 1 np.divide(0., 0.)
    

    NameError: log specified for invalid value (in divide) but no object with write method found.


# In[130]:

np.geterr()


# Out[130]:

#     {'divide': 'raise', 'invalid': 'warn', 'over': 'warn', 'under': 'ignore'}

# In[129]:

np.seterr(divide='raise', invalid='warn')


# Out[129]:

#     {'divide': 'raise', 'invalid': 'raise', 'over': 'warn', 'under': 'ignore'}

# In[87]:

5./0.


# Out[87]:


    ---------------------------------------------------------------------------
    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-87-cb534af67e91> in <module>()
    ----> 1 5./0.
    

    ZeroDivisionError: float division by zero


# In[39]:

type(specs)


# Out[39]:

#     list

# In[20]:

hyp.gethyperionbands().dtype


# Out[20]:

#     dtype([('Hyperion_Band', 'S7'), ('Average_Wavelength_nm', '<f8'), ('Full_Width_at_Half_the_Maximum_FWHM_nm', '<f8'), ('Spatial_Resolution_m', '<i8'), ('Not_Calibrated_X', 'S1')])

# In[23]:

hypsc.hyperionbands[hypsc.band_is_calibrated]


# Out[23]:

#     chararray(['band8', 'band9', 'band10', 'band11', 'band12', 'band13', 'band14',
#            'band15', 'band16', 'band17', 'band18', 'band19', 'band20',
#            'band21', 'band22', 'band23', 'band24', 'band25', 'band26',
#            'band27', 'band28', 'band29', 'band30', 'band31', 'band32',
#            'band33', 'band34', 'band35', 'band36', 'band37', 'band38',
#            'band39', 'band40', 'band41', 'band42', 'band43', 'band44',
#            'band45', 'band46', 'band47', 'band48', 'band49', 'band50',
#            'band51', 'band52', 'band53', 'band54', 'band55', 'band77',
#            'band56', 'band78', 'band57', 'band79', 'band80', 'band81',
#            'band82', 'band83', 'band84', 'band85', 'band86', 'band87',
#            'band88', 'band89', 'band90', 'band91', 'band92', 'band93',
#            'band94', 'band95', 'band96', 'band97', 'band98', 'band99',
#            'band100', 'band101', 'band102', 'band103', 'band104', 'band105',
#            'band106', 'band107', 'band108', 'band109', 'band110', 'band111',
#            'band112', 'band113', 'band114', 'band115', 'band116', 'band117',
#            'band118', 'band119', 'band120', 'band121', 'band122', 'band123',
#            'band124', 'band125', 'band126', 'band127', 'band128', 'band129',
#            'band130', 'band131', 'band132', 'band133', 'band134', 'band135',
#            'band136', 'band137', 'band138', 'band139', 'band140', 'band141',
#            'band142', 'band143', 'band144', 'band145', 'band146', 'band147',
#            'band148', 'band149', 'band150', 'band151', 'band152', 'band153',
#            'band154', 'band155', 'band156', 'band157', 'band158', 'band159',
#            'band160', 'band161', 'band162', 'band163', 'band164', 'band165',
#            'band166', 'band167', 'band168', 'band169', 'band170', 'band171',
#            'band172', 'band173', 'band174', 'band175', 'band176', 'band177',
#            'band178', 'band179', 'band180', 'band181', 'band182', 'band183',
#            'band184', 'band185', 'band186', 'band187', 'band188', 'band189',
#            'band190', 'band191', 'band192', 'band193', 'band194', 'band195',
#            'band196', 'band197', 'band198', 'band199', 'band200', 'band201',
#            'band202', 'band203', 'band204', 'band205', 'band206', 'band207',
#            'band208', 'band209', 'band210', 'band211', 'band212', 'band213',
#            'band214', 'band215', 'band216', 'band217', 'band218', 'band219',
#            'band220', 'band221', 'band222', 'band223', 'band224'], 
#           dtype='|S7')

# In[34]:

hypsc.band_is_calibrated


# Out[34]:

#     array([  426.82,   436.99,   447.17,   457.34,   467.52,   477.69,
#              487.87,   498.04,   508.22,   518.39,   528.57,   538.74,
#              548.92,   559.09,   569.27,   579.45,   589.62,   599.8 ,
#              609.97,   620.15,   630.32,   640.5 ,   650.67,   660.85,
#              671.02,   681.2 ,   691.37,   701.55,   711.72,   721.9 ,
#              732.07,   742.25,   752.43,   762.6 ,   772.78,   782.95,
#              793.13,   803.3 ,   813.48,   823.65,   833.83,   844.  ,
#              854.18,   864.35,   874.53,   884.7 ,   894.88,   905.05,
#              912.45,   915.23,   922.54,   925.41,   932.64,   942.73,
#              952.82,   962.91,   972.99,   983.08,   993.17,  1003.3 ,
#             1013.3 ,  1023.4 ,  1033.49,  1043.59,  1053.69,  1063.79,
#             1073.89,  1083.99,  1094.09,  1104.19,  1114.19,  1124.28,
#             1134.38,  1144.48,  1154.58,  1164.68,  1174.77,  1184.87,
#             1194.97,  1205.07,  1215.17,  1225.17,  1235.27,  1245.36,
#             1255.46,  1265.56,  1275.66,  1285.76,  1295.86,  1305.96,
#             1316.05,  1326.05,  1336.15,  1346.25,  1356.35,  1366.45,
#             1376.55,  1386.65,  1396.74,  1406.84,  1416.94,  1426.94,
#             1437.04,  1447.14,  1457.23,  1467.33,  1477.43,  1487.53,
#             1497.63,  1507.73,  1517.83,  1527.92,  1537.92,  1548.02,
#             1558.12,  1568.22,  1578.32,  1588.42,  1598.51,  1608.61,
#             1618.71,  1628.81,  1638.81,  1648.9 ,  1659.  ,  1669.1 ,
#             1679.2 ,  1689.3 ,  1699.4 ,  1709.5 ,  1719.6 ,  1729.7 ,
#             1739.7 ,  1749.79,  1759.89,  1769.99,  1780.09,  1790.19,
#             1800.29,  1810.38,  1820.48,  1830.58,  1840.58,  1850.68,
#             1860.78,  1870.87,  1880.98,  1891.07,  1901.17,  1911.27,
#             1921.37,  1931.47,  1941.57,  1951.57,  1961.66,  1971.76,
#             1981.86,  1991.96,  2002.06,  2012.15,  2022.25,  2032.35,
#             2042.45,  2052.45,  2062.55,  2072.65,  2082.75,  2092.84,
#             2102.94,  2113.04,  2123.14,  2133.24,  2143.34,  2153.34,
#             2163.43,  2173.53,  2183.63,  2193.73,  2203.83,  2213.93,
#             2224.03,  2234.12,  2244.22,  2254.22,  2264.32,  2274.42,
#             2284.52,  2294.61,  2304.71,  2314.81,  2324.91,  2335.01,
#             2345.11,  2355.21,  2365.2 ,  2375.3 ,  2385.4 ,  2395.5 ])

# In[20]:

len(bufferall)


# Out[20]:

#     11962

# In[22]:

bufferall[:10]


# Out[22]:

#     [(27, 154),
#      (27, 155),
#      (27, 156),
#      (27, 157),
#      (27, 158),
#      (27, 159),
#      (27, 160),
#      (27, 161),
#      (27, 162),
#      (27, 163)]

# In[24]:

len(specs_bufferall[:10][0])


# Out[24]:

#     97

# In[41]:

np.where(hyp.gethyperionbands().Not_Calibrated_X =='')[0]


# Out[41]:

#     array([  7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
#             20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
#             33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
#             46,  47,  48,  50,  52,  54,  56,  58,  60,  61,  62,  63,  64,
#             65,  67,  69,  71,  73,  75,  77,  79,  81,  83,  85,  87,  89,
#             91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
#            104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
#            117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
#            130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
#            143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
#            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
#            169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
#            182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
#            195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
#            208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
#            221, 222, 223])

# In[21]:

hypsc.hyperionbands[hypsc.band_is_calibrated]


# Out[21]:

#     chararray(['band8', 'band9', 'band10', 'band11', 'band12', 'band13', 'band14',
#            'band15', 'band16', 'band17', 'band18', 'band19', 'band20',
#            'band21', 'band22', 'band23', 'band24', 'band25', 'band26',
#            'band27', 'band28', 'band29', 'band30', 'band31', 'band32',
#            'band33', 'band34', 'band35', 'band36', 'band37', 'band38',
#            'band39', 'band40', 'band41', 'band42', 'band43', 'band44',
#            'band45', 'band46', 'band47', 'band48', 'band49', 'band50',
#            'band51', 'band52', 'band53', 'band54', 'band55', 'band77',
#            'band56', 'band78', 'band57', 'band79', 'band80', 'band81',
#            'band82', 'band83', 'band84', 'band85', 'band86', 'band87',
#            'band88', 'band89', 'band90', 'band91', 'band92', 'band93',
#            'band94', 'band95', 'band96', 'band97', 'band98', 'band99',
#            'band100', 'band101', 'band102', 'band103', 'band104', 'band105',
#            'band106', 'band107', 'band108', 'band109', 'band110', 'band111',
#            'band112', 'band113', 'band114', 'band115', 'band116', 'band117',
#            'band118', 'band119', 'band120', 'band121', 'band122', 'band123',
#            'band124', 'band125', 'band126', 'band127', 'band128', 'band129',
#            'band130', 'band131', 'band132', 'band133', 'band134', 'band135',
#            'band136', 'band137', 'band138', 'band139', 'band140', 'band141',
#            'band142', 'band143', 'band144', 'band145', 'band146', 'band147',
#            'band148', 'band149', 'band150', 'band151', 'band152', 'band153',
#            'band154', 'band155', 'band156', 'band157', 'band158', 'band159',
#            'band160', 'band161', 'band162', 'band163', 'band164', 'band165',
#            'band166', 'band167', 'band168', 'band169', 'band170', 'band171',
#            'band172', 'band173', 'band174', 'band175', 'band176', 'band177',
#            'band178', 'band179', 'band180', 'band181', 'band182', 'band183',
#            'band184', 'band185', 'band186', 'band187', 'band188', 'band189',
#            'band190', 'band191', 'band192', 'band193', 'band194', 'band195',
#            'band196', 'band197', 'band198', 'band199', 'band200', 'band201',
#            'band202', 'band203', 'band204', 'band205', 'band206', 'band207',
#            'band208', 'band209', 'band210', 'band211', 'band212', 'band213',
#            'band214', 'band215', 'band216', 'band217', 'band218', 'band219',
#            'band220', 'band221', 'band222', 'band223', 'band224'], 
#           dtype='|S7')

# In[44]:

np.array(specs)


# Out[44]:

#     array([[ 53.90000153,  49.77500153,  50.65000153, ...,   6.86250019,
#               6.80000019,   7.3499999 ],
#            [ 53.125     ,  51.07500076,  50.04999924, ...,   4.57499981,
#               4.11250019,   4.26249981],
#            [ 59.59999847,  53.65000153,  53.27500153, ...,   5.4000001 ,
#               5.9124999 ,   7.01249981],
#            ..., 
#            [ 86.90000153,  87.97499847,  89.09999847, ...,   5.13749981,
#               5.13749981,   6.1875    ],
#            [ 85.94999695,  87.07499695,  88.34999847, ...,   8.69999981,
#               8.46249962,   9.4375    ],
#            [ 86.94999695,  87.22499847,  88.02500153, ...,   4.98750019,
#               4.3375001 ,   4.61250019]], dtype=float32)

# In[45]:

_44.shape


# Out[45]:

#     (298, 198)

# In[ ]:

spec

