import pandas as pd, numpy as np, matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cols
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import colorsys

from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import mapclassify, pylab, colorsys
pd.set_option("display.precision", 3)

   
class Plot():
    
    def __init__(self, figsize, black_background, title):
    
        fig, ax = plt.subplots(1, figsize=figsize)

        # background black or white - basic settings
        rect = fig.patch 
        if black_background: 
            text_color = "white"
            rect.set_facecolor("black")
        else: 
            text_color = "black"
            rect.set_facecolor("white")
        
        font_size_primary = figsize[0]*1.50
        font_size_secondary = figsize[0]*1.25
        
        fig.suptitle(title, color = text_color, fontsize=font_size_primary, fontfamily = 'Times New Roman')
        fig.subplots_adjust(top=0.96)
        
        plt.axis("equal")
        self.fig, self.grid = fig, ax
        self.font_size_primary, self.font_size_secondary = font_size_primary, font_size_secondary
        self.text_color = text_color
                
class MultiPlot():
    
    def __init__(self, figsize, nrows, ncols, black_background, title = None):
    
        fig, grid = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

        rect = fig.patch 
        if black_background: 
            text_color = "white"
            rect.set_facecolor("black")
        else: 
            text_color = "black"
            rect.set_facecolor("white")
        
        font_size_primary = figsize[0]*1.35
        font_size_secondary = figsize[0]*1.15
        
        if title is not None:
            fig.suptitle(title, color = text_color, fontsize = font_size_secondary, fontfamily = 'Times New Roman', 
                         ha = 'center', va = 'center') 
            fig.subplots_adjust(top=0.92)
         
        self.fig, self.grid = fig, grid
        self.font_size_primary, self.font_size_secondary = font_size_primary, font_size_secondary
        self.text_color = text_color

def _single_plot(ax, gdf, column = None, scheme = None, bins = None, classes = 7, norm = None, cmap = None, color = 'red', alpha = 1.0, 
                legend = False, geometry_size = 1.0,  geometry_size_column = None, geometry_size_factor = None, zorder = 0):
    """
    It plots the geometries of a GeoDataFrame, coloring on the bases of the values contained in column, using a given scheme, on the provided Axes.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    ax: matplotlib.axes object
        the Axes on which plotting
    gdf: GeoDataFrame
        GeoDataFrame to be plotted 
    column: string
        Column on which the plot is based
    scheme: string
        classification method, choose amongst: https://pysal.org/mapclassify/api.html
    bins: list
        bins defined by the user
    classes: int
        classes when scheme is not "None"
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    cmap: string, matplotlib.colors.LinearSegmentedColormap
        see matplotlib colormaps for a list of possible values or pass a colormap
    color: string
        categorical color applied to all geometries when not using a column to color them
    alpha: float
        alpha value of the plotted layer
    legend: boolean
        if True, it shows the legend
    geometry_size: float
        markersize, when plotting a Point GeoDataFrame or linewidth when plotting a LineString GeoDataFrame
    geometry_size_column: string 
        name of the columnn, if any, of the GeoDataFrame whose values are to regulate the size of the geometries
    geometry_size_factor: float
        to control to what extent the values of the geometry_size_column impact the geometry_size
        For a Point GeoDataFrame, it rescales the geometry_size_column provided from 0 to 1 and applies the factor (e.g. rescaled variable's value [0-1] * factor).
    zorder: int   
        zorder of this layer; e.g. if 0, plots first, thus main GeoDataFrame on top; if 1, plots last, thus on top.
    """  
    
    gdf = gdf.copy()
    categorical = True
    if (column is not None): 
        if (gdf[column].dtype != 'O' ):
            gdf = gdf.reindex(gdf[column].abs().sort_values(ascending = True).index)
    
    # categorical map
    if (column is not None) & (scheme is None) & (norm is None) & (cmap is None): 
        cmap = rand_cmap(len(gdf[column].unique()))         
    
    if (norm is not None) | (scheme is not None):
        categorical = False
        color = None
        if cmap is None:
            cmap = kindlmann()
        
    if (column is not None) & (not categorical):
        if(gdf[column].dtype == 'O'):
            gdf[column] = gdf[column].astype(float)
        
    if bins is None: 
        c_k = {None}
        if classes is not None:
            c_k = {"k" : classes}
    else: 
        c_k = {'bins':bins, "k" : len(bins)}
        scheme = 'User_Defined'
    
    if gdf.iloc[0].geometry.geom_type == 'Point':
        if (geometry_size_factor is not None): 
            scaling_columnDF(gdf, geometry_size_column)
            gdf['geometry_size'] = np.where(gdf[geometry_size_column+'_sc'] >= 0.20, gdf[geometry_size_column+'_sc']*geometry_size_factor, 0.40) # marker size
            geometry_size = gdf['geometry_size']

        gdf.plot(ax = ax, column = column, markersize = geometry_size, categorical = categorical, color = color, scheme = scheme, cmap = cmap, 
                norm = norm, alpha = alpha, legend = legend, classification_kwds = c_k, zorder = zorder) 
        
    elif gdf.iloc[0].geometry.geom_type == 'LineString':
        if geometry_size_factor is not None:
            geometry_size = [(abs(value)*geometry_size_factor) if (abs(value)*geometry_size_factor) > 1.1 else 1.1 for value in gdf[geometry_size_column]]
        
        gdf.plot(ax = ax, column = column, categorical = categorical, color = color, linewidth = geometry_size, scheme = scheme, alpha = alpha, 
            cmap = cmap, norm = norm, legend = legend, classification_kwds = c_k, capstyle = 'round', joinstyle = 'round', zorder = zorder) 
                
    else:
        gdf.plot(ax = ax, column = column, categorical = categorical, color = color, scheme = scheme, edgecolor = 'none', alpha = alpha, cmap = cmap,
            norm = norm, legend = legend, classification_kwds = c_k, zorder = zorder)       
        
 
def plot_gdf(gdf, column = None, title = None, black_background = True, figsize = (15,15), scheme = None, bins = None, classes = None, norm = None,
            cmap = None, color = None, alpha = None, legend = False, geometry_size = 1.0, geometry_size_column = None, 
            geometry_size_factor = None, cbar = False, cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrinkage = 0.75,
            axes_frame = False, base_map_gdf = pd.DataFrame({"a" : []}), base_map_color = None, base_map_alpha = 0.4, base_map_geometry_size = 1.1,  
            base_map_zorder = 0):

    """
    It plots the geometries of a GeoDataFrame, coloring on the bases of the values contained in column, using a given scheme.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        GeoDataFrame to be plotted 
    column: string
        Column on which the plot is based
    title: string 
        title of the plot
    black_background: boolean 
        black background or white
    fig_size: float
        size of the figure's side extent
    scheme: string
        classification method, choose amongst: https://pysal.org/mapclassify/api.html
    bins: list
        bins defined by the user
    classes: int
        number of classes for categorising the data when scheme is not "None"
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    cmap: string, matplotlib.colors.LinearSegmentedColormap
        see matplotlib colormaps for a list of possible values or pass a colormap
    color: string
        categorical color applied to all geometries when not using a column to color them
    alpha: float
        alpha value of the plotted layer
    legend: boolean
        if True, show legend, otherwise don't
    cbar: boolean
        if True, show colorbar, otherwise don't; when True it doesn't show legend
    cbar_ticks: int
        number of ticks along the colorbar
    cbar_max_symbol: boolean
        if True, it shows the ">" next to the highest tick's label in the colorbar (useful when normalising)
    cbar_min_max: boolean
        if True, it only shows the labels of the lowest and highest ticks of the colorbar
    cbar_shrink: float
        fraction by which to multiply the size of the colorbar
    axes_frame: boolean
        if True, it shows the axes' frame
    geometry_size: float
        markersize, when plotting a Point GeoDataFrame or linewidth when plotting a LineString GeoDataFrame
    geometry_size_column: string 
        name of the columnn, if any, of the GeoDataFrame whose values are to regulate the size of the geometries
    geometry_size_factor: float
        to control to what extent the values of the geometry_size_column impact the geometry_size
        For a Point GeoDataFrame, it rescales the geometry_size_column provided from 0 to 1 and applies the factor (e.g. rescaled variable's value [0-1] * factor).
    base_map_gdf: GeoDataFrame
        a desired additional layer to use as a base map        
    base_map_color: string
        color applied to all geometries of the base map
    base_map_alpha: float
        base map's alpha value
    base_map_geometry_size: float
        base map's marker size when the base map is a Point GeoDataFrame

    base_map_zorder: int   
        zorder of the layer; e.g. if 0, plots first, thus main GeoDataFrame on top; if 1, plots last, thus on top.
        
    Returns
    -------
    fig: matplotlib.figure.Figure object
        the resulting figure
    """   
    
    # fig,ax set up
    plot = Plot(figsize = figsize, black_background = black_background, title = title)
    fig, ax = plot.fig, plot.grid
    

    _set_axes_frame(axes_frame, ax, black_background, plot.text_color)
    ax.set_aspect("equal")
    
    zorder = 0
    if (not base_map_gdf.empty):
        _plot_baseMap(gdf = base_map_gdf, ax = ax, color = base_map_color, geometry_size = base_map_geometry_size, alpha = base_map_alpha, 
            zorder = base_map_zorder )
        if base_map_zorder == 0:
            zorder = 1
   
    if geometry_size_column is None:
        geometry_size_column = column
    
    _single_plot(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, alpha = alpha, 
                geometry_size = geometry_size, geometry_size_column = geometry_size_column, geometry_size_factor = geometry_size_factor, 
                zorder = zorder, legend = legend)

    if legend: 
        _generate_legend(plot, ax, black_background) 
    elif cbar:
        if norm is None:
            min_value = gdf[column].min()
            max_value = gdf[column].max()
            norm = plt.Normalize(vmin = min_value, vmax = max_value)
            
        _generate_colorbar(plot, cmap, norm = norm, ticks = cbar_ticks, symbol = cbar_max_symbol, min_max = cbar_min_max, shrinkage = cbar_shrinkage)
    
    return fig    
                      
def plot_gdfs(list_gdfs = [], column = None, ncols = 2, main_title = None, titles = [], black_background = True, figsize = (15,30), scheme = None, 
                bins = None, classes = None, norm = None, cmap = None, color = None, alpha = None, legend = False, cbar = False, 
                cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrinkage = 0.75, axes_frame = False, 
                geometry_size = None, geometry_size_column = None, geometry_size_factor = None): 
                     
    """
    It plots the geometries of a list of GeoDataFrame, containing the same type of geometry. Coloring is based on a provided column (that needs to 
    be a column in each passed GeoDataFrame), using a given scheme.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    list_gdfs: list of GeoDataFrames
        GeoDataFrames to be plotted
    column: string
        Column on which the plot is based
    main_title: string 
        main title of the plot
    titles: list of string
        list of titles to be assigned to each quadrant (axes) of the grid
    black_background: boolean 
        black background or white
    fig_size: float
        size figure extent
    scheme: string
        classification method, choose amongst: https://pysal.org/mapclassify/api.html
    bins: list
        bins defined by the user
    classes: int
        number of classes for categorising the data when scheme is not "None"
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    cmap: string, matplotlib.colors.LinearSegmentedColormap
        see matplotlib colormaps for a list of possible values or pass a colormap
    color: string
        categorical color applied to all geometries when not using a column to color them
    alpha: float
        alpha value of the plotted layer
    legend: boolean
        if True, show legend, otherwise don't
    cbar: boolean
        if True, show colorbar, otherwise don't; when True it doesn't show legend
    cbar_ticks: int
        number of ticks along the colorbar
    cbar_max_symbol: boolean
        if True, it shows the ">" next to the highest tick's label in the colorbar (useful when normalising)
    cbar_min_max: boolean
        if True, it only shows the labels of the lowest and highest ticks of the colorbar
    cbar_shrink: float
        fraction by which to multiply the size of the colorbar
    axes_frame: boolean
        if True, it shows the axes' frame
    geometry_size: float
        markersize, when plotting a Point GeoDataFrame or linewidth when plotting a LineString GeoDataFrame
    geometry_size_column: string 
        name of the columnn, if any, of the GeoDataFrame whose values are to regulate the size of the geometries
    geometry_size_factor: float
        to control to what extent the values of the geometry_size_column impact the geometry_size
        For a Point GeoDataFrame, it rescales the geometry_size_column provided from 0 to 1 and applies the factor (e.g. rescaled variable's value [0-1] * factor).
        
    
    Returns
    -------
    fig: matplotlib.figure.Figure object
        the resulting figure
    """              
                     
    if ncols == 2:
        nrows, ncols = int(len(list_gdfs)/2), 2
        if (len(list_gdfs)%2 != 0): 
            nrows = nrows+1
    else:
        nrows, ncols = int(len(list_gdfs)/3), 3
        if (len(list_gdfs)%3 != 0): 
            nrows = nrows+1

    multiPlot = MultiPlot(figsize = figsize, nrows = nrows, ncols = ncols, black_background = black_background, 
                          title = main_title)
    
    fig, grid = multiPlot.fig, multiPlot.grid   
    
    if nrows > 1: 
        grid = [item for sublist in grid for item in sublist]
    if cbar:
        legend = False
        if (norm is None):
            min_value = min([gdf[column].min() for gdf in list_gdfs])
            max_value = max([gdf[column].max() for gdf in list_gdfs])
            norm = plt.Normalize(vmin = min_value, vmax = max_value)
    
    if nrows > 1: 
        axes = [item for sublist in grid for item in sublist]
    else:
        axes = grid
    
    legend_ax = False
    for n, ax in enumerate(axes):
        _set_axes_frame(axes_frame, ax, black_background, multiPlot.text_color)    
        ax.set_aspect("equal")
        if n > len(list_gdfs)-1: 
            continue # when odd nr of gdfs    
        
        gdf = list_gdfs[n]
        if len(titles) > 0:
            ax.set_title(titles[n], loc='center', fontfamily = 'Times New Roman', fontsize = multiPlot.font_size_primary, color = multiPlot.text_color, pad = 15)
                   
        geometry_size_column = column
        if geometry_size_columns != []:
            geometry_size_column = geometry_size_columns[n]
        
        if (legend) & (ax == axes[-1]):
            legend_ax = True
        
        _single_plot(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, 
                    alpha = alpha, legend = legend_ax, geometry_size = geometry_size, geometry_size_column = geometry_size_column, 
                    geometry_size_factor = geometry_size_factor)
                    
    if legend:
        _generate_legend(multiPlot, ax, black_background)
    elif cbar:
        _set_colorbar(multiPlot, cmap, norm = norm, ticks = cbar_ticks, symbol = cbar_max_symbol, min_max = cbar_min_max, 
                    shrinkage = cbar_shrinkage)
            
    return fig
   
def plot_gdf_grid(gdf = None, columns = [], ncols = 2, titles = [], black_background = True, figsize = (15,15), scheme = None, bins = None, 
                classes = None, norm = None, cmap = None, color = None, alpha = None, legend = False, cbar = False, 
                cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrinkage = 0.75, axes_frame = False, geometry_size = None, 
                geometry_size_columns = [], geometry_size_factor = None): 
    """
    It plots the geometries of a GeoDataFrame, coloring on the bases of the values contained in the provided columns, using a given scheme.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        GeoDataFrame to be plotted 
    column: string
        Column on which the plot is based
    title: string 
        title of the plot
    black_background: boolean 
        black background or white
    fig_size: float
        size figure extent
    scheme: string
        classification method, choose amongst: https://pysal.org/mapclassify/api.html
    bins: list
        bins defined by the user
    classes: int
        number of classes for categorising the data when scheme is not "None"
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    cmap: string, matplotlib.colors.LinearSegmentedColormap
        see matplotlib colormaps for a list of possible values or pass a colormap
    color: string
        categorical color applied to all geometries when not using a column to color them
    alpha: float
        alpha value of the plotted layer
    legend: boolean
        if True, show legend, otherwise don't
    cbar: boolean
        if True, show colorbar, otherwise don't; when True it doesn't show legend
    cbar_ticks: int
        number of ticks along the colorbar
    cbar_max_symbol: boolean
        if True, it shows the ">" next to the highest tick's label in the colorbar (useful when normalising)
    cbar_min_max: boolean
        if True, it only shows the labels of the lowest and highest ticks of the colorbar
    cbar_shrink: float
        fraction by which to multiply the size of the colorbar
    axes_frame: boolean
        if True, it shows the axes' frame
    geometry_size: float
        markersize, when plotting a Point GeoDataFrame or linewidth when plotting a LineString GeoDataFrame
    geometry_size_columns: List 
        List of the name of the columnn, if any, of the passed GeoDataFrames whose values are to regulate the size of the geometries
    geometry_size_factor: float
        to control to what extent the values of the geometry_size_column impact the geometry_size
        For a Point GeoDataFrame, it rescales the geometry_size_column provided from 0 to 1 and applies the factor (e.g. rescaled variable's value [0-1] * factor).
    geometry_size_factor: float
        when provided, it rescales the column provided, if any, from 0 to 1 and it uses the geometry_size_factor to rescale the line 
        width accordingly 
        (e.g. rescaled variable's value [0-1] * factor), when plotting a LineString GeoDataFrame
    """   
    
    if ncols == 2:
        nrows, ncols = int(len(columns)/2), 2
        if (len(columns)%2 != 0): 
            nrows = nrows+1
    else:
        nrows, ncols = int(len(columns)/3), 3
        if (len(columns)%3 != 0): 
            nrows = nrows+1
            
    multiPlot = MultiPlot(figsize = figsize, nrows = nrows, ncols = ncols, black_background = black_background)
    fig, grid = multiPlot.fig, multiPlot.grid   
    
    legend_ax = False
    if cbar:
        legend = False
        if norm is None:
            min_value = min([gdf[column].min() for column in columns])
            max_value = max([gdf[column].max() for column in columns])
            norm = plt.Normalize(vmin = min_value, vmax = max_value)
    
    if nrows > 1: 
        axes = [item for sublist in grid for item in sublist]
    else:
        axes = grid
        
    for n, ax in enumerate(axes):
        ax.set_aspect("equal")
        _set_axes_frame(axes_frame, ax, black_background, multiPlot.text_color)
        
        if n > len(columns)-1: 
            continue # when odd nr of columns
        
        column = columns[n]
        if len(titles) > 0:          
            ax.set_title(titles[n],loc='center', fontfamily = 'Times New Roman', fontsize = multiPlot.font_size_primary, color = multiPlot.text_color, pad = 15)
                
        geometry_size_column = column
        if geometry_size_columns != []:
            geometry_size_column = geometry_size_columns[n]
        if (legend) & (ax == axes[-1]):
            legend_ax = True
        
        _single_plot(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, 
                    alpha = alpha, legend = legend_ax, geometry_size = geometry_size, geometry_size_column = geometry_size_column, 
                    geometry_size_factor = geometry_size_factor)
        
    if legend:
        _generate_legend(multiPlot, ax, black_background)
    elif cbar:   
        _generate_colorbar(plot = multiPlot, cmap = cmap, norm = norm, ticks = cbar_ticks, symbol = cbar_max_symbol, min_max = cbar_min_max, 
                    shrinkage = cbar_shrinkage)

    return fig

def _plot_baseMap(gdf = None, ax = None, color = None, geometry_size = None, alpha = 0.5, zorder = 0):
    
    if gdf.iloc[0].geometry.geom_type == 'LineString':
        gdf.plot(ax = ax, color = color, linewidth = geometry_size, alpha = alpha,zorder = zorder)
    if gdf.iloc[0].geometry.geom_type == 'Point':
        gdf.plot(ax = ax, color = color, markersize = geometry_size, alpha = alpha, zorder = zorder)
    if gdf.iloc[0].geometry.geom_type == 'Polygon':
        gdf.plot(ax = ax, color = color, alpha = alpha, zorder = zorder)
    

def plot_multiplex(M, multiplex_edges):

    node_Xs = [float(node["x"]) for node in M.nodes.values()]
    node_Ys = [float(node["y"]) for node in M.nodes.values()]
    node_Zs = np.array([float(node["z"])*2000 for node in M.nodes.values()])
    node_size = []
    size = 1
    node_color = []

    for i, d in M.nodes(data=True):
        if d["station"]:
            node_size.append(9)
            node_color.append("#ec1a30")
        elif d["z"] == 1:
            node_size.append(0.0)
            node_color.append("#ffffcc")
        elif d["z"] == 0:
            node_size.append(8)
            node_color.append("#ff8566")

    lines = []
    line_width = []
    geometry_sizeidth = 0.4
    
    # edges
    for u, v, data in M.edges(data=True):
        xs, ys = data["geometry"].xy
        zs = [M.node[u]["z"]*2000 for i in range(len(xs))]
        if data["layer"] == "intra_layer": 
            zs = [0, 2000]
        
        lines.append([list(a) for a in zip(xs, ys, zs)])
        if data["layer"] == "intra_layer": 
            line_width.append(0.2)
        elif data["pedestrian"] == 1: 
            line_width.append(0.1)
        else: 
            line_width.append(geometry_sizeidth)

    fig_height = 40
    lc = Line3DCollection(lines, linewidths=line_width, alpha=1, color="#ffffff", zorder=1)

    west, south, east, north = multiplex_edges.total_bounds
    bbox_aspect_ratio = (north - south) / (east - west)*1.5
    fig_width = fig_height +90 / bbox_aspect_ratio/1.5
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection="3d")
    ax.add_collection3d(lc)
    ax.scatter(node_Xs, node_Ys, node_Zs, s=node_size, c=node_color, zorder=2)
    ax.set_ylim(south, north)
    ax.set_xlim(west, east)
    ax.set_zlim(0, 2500)
    ax.axis("off")
    ax.margins(0)
    ax.tick_parageometry_size(which="both", direction="in")
    fig.canvas.draw()
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    return(fig)
       
def _generate_legend(plot, ax, black_background):
    """ 
    It generate the legend for a figure.
    
    Parameters
    ----------
    ax: matplotlib.axes object
        the Axes on which plotting
    text_color: string
        the text color
    font_size: int
        the legend's labels text size
    """
    leg = ax.get_legend()
    for handle in leg.legendHandles:   
        if not isinstance(handle, mpl.lines.Line2D):
            handle._legmarker.set_markersize(plot.fig.get_size_inches()[0]*0.90)
  
    final_legend = plot.fig.legend(handles = leg.legendHandles, labels = [t.get_text() for t in leg.texts], loc = 'center right',
                bbox_to_anchor = (1.10, 0.5))

    if black_background:
        text_color = 'black'
    else: 
        text_color = 'white'

    plt.setp(final_legend.texts, family= 'Times New Roman', fontsize = plot.font_size_secondary, color = text_color, va = 'center')
    final_legend.get_frame().set_linewidth(0.0) # remove legend border
    final_legend.set_zorder(102)
    
    if not black_background:
        final_legend.get_frame().set_facecolor('black')
        final_legend.get_frame().set_alpha(0.90)  
    else:
        final_legend.get_frame().set_facecolor('white')
        final_legend.get_frame().set_alpha(0.75)  
    
    leg.remove()
    plot.fig.add_artist(final_legend)         
    
def _generate_colorbar(plot = None, cmap = None, norm = None, ticks = 5, symbol = False, min_max = False, shrinkage = 0.95):
    """ 
    It plots a colorbar, given some settings.
    
    Parameters
    ----------
    fig: matplotlib.figure.Figure
        The figure container for the current plot
    pos: list of float
        the axes positions
    sm: matplotlib.cm.ScalarMappable
        a mixin class to map scalar data to RGBA
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    text_color: string
        the text color
    font_size: int
        the colorbar's labels text size
    ticks: int
        the number of ticks along the colorbar
    symbol: boolean
        if True, it shows the ">" next to the highest tick's label in the colorbar (useful when normalising)
    cbar_min_max: boolean
        if True, it only shows the ">" and "<" as labels of the lowest and highest ticks' the colorbar
    """
        
    cb = plot.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plot.grid, shrink = shrinkage)
    tick_locator = ticker.MaxNLocator(nbins=ticks)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.outline.set_visible(False)

    ticks = list(cb.get_ticks())
    for t in ticks: 
        if (t == ticks[-1]) & (t != norm.vmax) :
            ticks[-1] = norm.vmax

    if min_max:
        ticks = [norm.vmin, norm.vmax]
    
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([round(t,1) for t in ticks])
    if symbol:
        cb.ax.set_yticklabels([round(t,1) if t < norm.vmax else "> "+str(round(t,1)) for t in cb.ax.get_yticks()])

    plt.setp(plt.getp(cb.ax, "yticklabels"), color = plot.text_color, fontfamily = 'Times New Roman', fontsize= plot.font_size_secondary)

def _set_axes_frame(axes_frame = False, ax = None, black_background = False, text_color = 'black'):
    """ 
    It draws the axis frame.
    
    Parameters
    ----------
    ax: matplotlib.axes
        the Axes on which plotting
    black_background: boolean
        it indicates if the background color is black
    text_color: string
        the text color
    """
    if not axes_frame:
        ax.set_axis_off()
        return
      
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.tick_params(axis= 'both', which= 'both', length=0)
    
    for spine in ax.spines:
        ax.spines[spine].set_color(text_color)
    if black_background: 
        ax.set_facecolor('black')
             
def normalize(n, range1, range2):

    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]           
            
# Generate random colormap
def rand_cmap(nlabels, type_color ='soft'):
    """ 
    It generates a categorical random color map, given the number of classes
    
    Parameters
    ----------
    nlabels: int
        the number of categories to be coloured 
    type_color: string {"soft", "bright"} 
        it defines whether using bright or soft pastel colors, by limiting the RGB spectrum
       
    Returns
    -------
    cmap: matplotlib.colors.LinearSegmentedColormap
        the color map
    """   
    if type_color not in ('bright', 'soft'):
        type_color = 'bright'
    
    # Generate color map for bright colors, based on hsv
    if type_color == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=0.8),
                          np.random.uniform(low=0.2, high=0.8),
                          np.random.uniform(low=0.9, high=1.0)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))


        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type_color == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap

def kindlmann():
    """ 
    It returns a Kindlmann color map. See https://ieeexplore.ieee.org/document/1183788
       
    Returns
    -------
    cmap: matplotlib.colors.LinearSegmentedColormap
        the color map
    """   

    kindlmann_list = [(0.00, 0.00, 0.00,1), (0.248, 0.0271, 0.569, 1), (0.0311, 0.258, 0.646,1),
            (0.019, 0.415, 0.415,1), (0.025, 0.538, 0.269,1), (0.0315, 0.658, 0.103,1),
            (0.331, 0.761, 0.036,1),(0.768, 0.809, 0.039,1), (0.989, 0.862, 0.772,1),
            (1.0, 1.0, 1.0)]
    cmap = LinearSegmentedColormap.from_list('kindlmann', kindlmann_list)
    return cmap
              
def lighten_color(color, amount=0.5):
    """
    This function can be found here https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a, author: Ian Hincks.
    
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = cols.cnames[color]
    except:
        c = color
    
    c = colorsys.rgb_to_hls(*cols.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    
def scaling_columnDF(df, column, inverse = False):
    """
    It rescales the values in a dataframe's columns from 0 to 1
    
    Parameters
    ----------
    df: pandas DataFrame
        a DataFrame
    column: string
        the column name, representing the column to rescale
    inverse: boolean
        if true, rescales from 1 to 0 instead of 0 to 1
    ----------
    """
    
    df[column+"_sc"] = (df[column]-df[column].min())/(df[column].max()-df[column].min())
    if inverse: 
        df[column+"_sc"] = 1-(df[column]-df[column].min())/(df[column].max()-df[column].min())
        