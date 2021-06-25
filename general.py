import matplotlib as mpl
import pickle

def mpl_settings():
    """ Set personal preferences for the matplotlib package"""
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'sans-serif' : ['Helvetica','Arial','DevaVu Sans'],
            'size' : 18
            }
    mpl.rc('font', **font)
    mpl.rc('axes', titlepad=20)
    mpl.rc('svg', fonttype='none')
    mpl.rc('figure', facecolor = 'w')
    mpl.rc('lines', linewidth = 3.0)
    mpl.rc('xtick', direction='in')
    mpl.rc('ytick', direction='in')
    mpl.rc('legend', frameon=False, fontsize=14)
    
def save_figure(fig,filename):
    """
    Save a matplotlib figure as an EPS, PDF and PNG.
    The filename should be input as a path (which can be relative) without an
    extension.
    
    Examples
    ========
    save_figure(fig,'fig_file')
    save_figure(fig,'C:/folder/fig_file')
    save_figure(fig,'../output/fig_file')
    """
    fig.savefig(filename+'.eps')
    fig.savefig(filename+'.pdf')
    fig.savefig(filename+'.png')
    fig.savefig(filename+'.svg')
    
def pickle_save(filepath,args):
    """
    Save the variables specified by args in pickle file specified by filepath.
    
    Examples
    ========
    pickle_save('file.pkl',variable)
    pickle_save('../folder/file.pkl',(var1,var2))
    """
    with open(filepath, 'wb') as f:
        pickle.dump(args, f)
      
def pickle_load(filepath):
    """
    Returns the variables stored within the pickle serialisation stored within
    filepath.
    """
    with open(filepath, 'rb') as f:
       return pickle.load(f)