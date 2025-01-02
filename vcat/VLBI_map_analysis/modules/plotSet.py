from fractions import Fraction
import sys
from tempfile import NamedTemporaryFile
from matplotlib.image import imread
'''
To get the ideal image size I combined codes from

https://jwalton.info/Embed-Publication-Matplotlib-Latex/

and

https://kavigupta.org/2019/05/18/Setting-the-size-of-figures-in-matplotlib/
'''
def get_size(fig, dpi=100):
    with NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name, bbox_inches='tight', dpi=dpi)
        height, width, _channels = imread(f.name).shape
        return width / dpi, height / dpi

def set_size(width, fraction=1,subplots=(1,1),ratio=False):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Taken from https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
       Width in pts, or string of predined document type
    fraction: float,optional
       Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
    The number of rows and columns of subplots

    Returns
    -------
    fig_dim: tuple
       Dimensions of figure in inches
    """
    if width.find('_')!=-1:
        w               = width.split('_')
        width       = w[0]
        fraction= float(w[1])
    if width =='aanda':
        width_pt = 256.0748
    elif width =='aanda*':
        width_pt = 523.5307
    elif width == 'beamer':
        width_pt = 342
    elif width == 'screen':
        width_pt = 600
    else:
        width_pt = width
    # Width of figure
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2.
    if not ratio:
        ratio = golden_ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio* (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def set_scaled_size():

    return

def set_corrected_size(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
    target_width, target_height = size
    set_width, set_height = target_width, target_height # reasonable starting point
    deltas = [] # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False
