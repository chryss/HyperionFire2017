# library of auxiliary functions for this project

import pickle

def xy2ji(xytup, geotiff):
    """Turn an (x, y) coordinate pair into a (j, i) col-row index pair
    """
    restup = geotiff.xy2ij(*xytup)
    return (restup[1], restup[0])

def specstopickle(outfilepath, specs):
    with open(outfilepath, "w") as sink:
        pickle.dump(specs, sink)
