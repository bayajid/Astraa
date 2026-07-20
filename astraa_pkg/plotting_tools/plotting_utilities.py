from matplotlib import cm, scale
from matplotlib import colors as mcolors
import matplotlib as mpl
from mycolorpy import colorlist as mcp

markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

def give_color_range(nr_colors):
    cmap = cm.rainbow
    # range_normalizer = mpl.colors.Normalize(vmin=0, vmax=nr_colors, clip = True)
    norm = mcolors.Normalize(vmin=0, vmax=nr_colors, clip = True)
    colors = [cmap(norm(ii)) for ii in list(range(nr_colors))]
    return colors
def give_color_list(y_dat_list, c_option = 0):
    """return list of colors for multi-line plots

    Args:
        y_dat_list (list of plotted lines): _description_
        c_option (int, optional): index for colors. Defaults to 0.

    Returns:
        color_list: list of color codes, to be indexed by 0-max int
    """    
    colormaps = ['coolwarm', 'bwr', 'seismic', 'gnuplot', 'brg', 'jet', 'turbo']
    color_list=mcp.gen_color_normalized(cmap=colormaps[c_option],data_arr=y_dat_list)
    return color_list