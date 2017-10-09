from __future__ import print_function
# library of auxiliary functions for this project
try:
    import h5py
except ImportError:
    print("HDF5 library is not available on this system.")
import pickle
 
def xy2ji(xytup, geotiff):
    """Turn an (x, y) coordinate pair into a (j, i) col-row index pair
    """
    restup = geotiff.xy2ij(*xytup)
    return (restup[1], restup[0])

def specstopickle(outfilepath, specs):
    import pickle
    """Pickle something, for example a list of spectra
    """
    with open(outfilepath, "w") as sink:
        pickle.dump(specs, sink)

def specsfrompickle(filepath, hypsc, ii, jj, picklefile=True, calculate=False):
    """Load spectra from pickle file. Optionally save them to pickle. 
    """
    if picklefile:
        try:        
            with open(filepath, "rU") as source:
                specs = pickle.load(source)
        except IOError, EOFError:
            calculate = True
    if (calculate or not picklefile):
        specs = [hypsc.spectrum(i, j) for i, j in itertools.izip(ii, jj)]
        if picklefile:
            specstopickle(filepath, specs)
    return specs

def specsfromhdf5(filepath, startbandidx=0):
    """Load spectra from HDF5 file
    """
    with h5py.File(filepath, "r") as infh:
        specs = infh['spectrum'][:, startbandidx:]
        ii = infh['i_row_idx'][...]
        jj = infh['j_col_idx'][...]
        wav = infh['bandwavelenght_nm'][startbandidx:]
        bn = infh['bandname'][startbandidx:]
        return specs, ii, jj, wav, bn

