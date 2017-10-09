"""
configuration for Hyperion processing

last edited: 2015-06-03
Chris Waigl - cwaigl@alaska.edu
"""

import os.path

# paths
BASEDIR = "/Volumes/SCIENCE_mobile_mac/Fire/DATA_BY_PROJECT/HyspIRI_Prep_2014/"
SCRIPTOUTINFIX = 'scriptout'
GISPOSTFIX = 'GIS'
DIR_boundary = "2004Boundary"
DIR_crazy = "2004Crazy"
DIR_2009 = "2009WoodRiver"
HYPSCENE_crazy = "EO1H0680132004192110KY"
HYPSCENE_boundary = "EO1H0690142004201110PX"
HYPSCENE_2009 = "EO1H0690142009214110KF"

# bands etc.
# band to be used as a high-SWIR default for initializing fire pixels
BANDNUM = 210

# fire stuff
FIRE = 'boundary'
FIREPARAM = {
    'boundary': {
        'infix': '_CLIP',
        'filedir': DIR_boundary,
        'sceneid': HYPSCENE_boundary,
        'firename': 'boundary',
        'fireyear': 2004,
        'figdim': (15, 19),
        'colbarpos': [0.125, 0.2, 0.725, 0.025],
    },
    'crazy': {
        'infix': '_CLIP',
        'filedir': DIR_crazy,
        'sceneid': HYPSCENE_crazy,
        'firename': 'crazy',
        'fireyear': 2004,
        'figdim': (15, 19),
        'colbarpos': [0.125, 0.2, 0.725, 0.025],
    },
    'woodriver': {
        'infix': '_CLIP',
        'filedir': DIR_2009,
        'sceneid': HYPSCENE_2009,
        'firename': 'woodriver',
        'fireyear': 2009,
        'figdim': (15, 25),
        'colbarpos': [0.125, 0.125, 0.74, 0.02],
    },
}

FIREPOINTSFILE = os.path.join(
    BASEDIR,
    FIREPARAM[FIRE]['filedir'],
    GISPOSTFIX,
    'firepixels_b{}_{}.shp'.format(
        BANDNUM,
        FIREPARAM[FIRE]['firename']
    )
)
BUFFERFILE = os.path.join(
    BASEDIR,
    FIREPARAM[FIRE]['filedir'],
    GISPOSTFIX,
    'circbuffer_b{}_{}.shp'.format(
        BANDNUM,
        FIREPARAM[FIRE]['firename']
    )
)
SCENEFILE = os.path.join(
    BASEDIR,
    FIREPARAM[FIRE]['filedir'],
    FIREPARAM[FIRE]['sceneid']
)

# geometry choices

RADTHRESH = 5.0  # Radiance threshold.
                 # In the SWIR, non-fire emitted and reflected radiances
                 # are much smaller.
LARGEBUFF = 100 # in m
