import pandas as pd, numpy as np, geopandas as gpd
import cityImage as ci
import math
from shapely.geometry import*
from shapely.ops import*
import ast
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def aggregate_runs(df_list, route_choice_models, edges_gdf, ddof = 0):
    """
    This function aggregates a series of dataframes generated by the ABM's runs.
    Per each passed route choice model, it extracts the volumes per street segment in each run, computes the median, mean and SD.
    Finally, it attaches such information, to the edges_gdf file so to associate directly the volumes with the street segment.
     
    Parameters
    ----------
    df_list: List of Pandas DataFrame
        A list contianing a dataframe per each run with the corresponding volumes
    route_choice_models: List of String
        The list of the abbreviation of the route choice models
    edges_gdf: LineString GeoDataFrame
        The GeoDataFrame of the edges of a street network
    ddof: int
        the Delta Degrees of Freedom for computing the standard deviation
    
    Returns
    -------
    LineString GeoDataFrame
    """

    stat_runs = pd.DataFrame(index= df_list[0].index, columns=['edgeID'] + route_choice_models)
    stat_runs['edgeID'] = df_list[0].edgeID
    stat_runs.index = stat_runs['edgeID']

    for n, df in enumerate(df_list):
        df.index = df['edgeID']
        tmp = df[[col for col in df if col.startswith(route_choice_models[n])]] # only run values
        df['median'] = tmp.median(axis = 1)
        df['mean'] = tmp.mean(axis = 1)
        df['std'] = tmp.std(axis = 1, ddof = ddof)

        stat_runs[route_choice_models[n]] = df["median"]
        stat_runs[route_choice_models[n]+'_std'] = df['std']
        stat_runs[route_choice_models[n]+'_mean'] = df['mean']

    stat_runs.index.name = None
    edges_gdf = pd.merge(edges_gdf, stat_runs, left_on = "edgeID", right_on = "edgeID", how = 'left')
    edges_gdf.index = edges_gdf.edgeID
    edges_gdf.index.name = None
    
    return edges_gdf

def get_edgesID(row, columns):
    """
    The ABM stores the sequence of edgeIDs traversed in each route in different columns, given the 254 characters limit per field.
    This functions merges such lists in one List.
     
    Parameters
    ----------
    row: Pandas Series
        A row of the GeoDataFrame containing the routes
    columns: List of String
        A routes GeoDataFrame's columns
        
    Returns
    -------
    List of Integer
    """
    
    edgeID_string = row['edgeIDs_0']
    counter = 1
    while True:
        if 'edgeIDs_'+str(counter) not in columns:
            break
        if row['edgeIDs_'+str(counter)] is None:
            break
        edgeID_string = edgeID_string + row['edgeIDs_'+str(counter)]
        counter += 1

    return ast.literal_eval(edgeID_string)

   
def compute_deviation_from(edges_gdf, route_choice_models, comparison, max_err):
    """
    It computes the deviation of the volumes emerging from a series of scenarios (route_choice_models) from a term of comparison (e.g. observational counts, another scenario volumes).
    The term of comparison has to be a column of values in the edges_gdf GeoDataFrame.
     
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The GeoDataFrame of the edges of a street network
    route_choice_models: List of String
        The list of the abbreviations of the route choice models
    comparison: String
        The name of the column used for the comparison with the ABM volumes
    max_err: int
        The max standard deviation error to use as a bound
    
    Returns
    -------
    LineString GeoDataFrame
    """

    for n, column in enumerate(route_choice_models): 
        edges_gdf[column+'_std_err'] = 0
        edges_gdf[column+'_diff'] = 0

        for row in edges_gdf.itertuples():
            median = edges_gdf.loc[row[0]][column] #column = median of the criterion considered
            stdv = edges_gdf.loc[row[0]][column+'_std']
            
            diff = median-edges_gdf.loc[row[0]][comparison]

            if (median == 0) & (stdv == 0) & (edges_gdf.loc[row[0]][comparison] == 0): std_err = 0
            elif (stdv == 0) & (diff > 0): std_err = max_err
            elif (stdv == 0) & (diff < 0): std_err = -max_err
            elif (stdv == 0) & (diff == 0): std_err = 0
            else: std_err = diff/stdv

            # set columns
            edges_gdf.at[row[0], column+'_std_err'] = std_err
            edges_gdf.at[row[0], column+'_diff'] = abs(diff)
        
    return edges_gdf

def find_tracks_OD(index, track_geometry, unionStreet, nodes_gdf):
    """
    It determines the origin and destination of a GPS track.
    This should be cleaned and snapped to the street network.
    This function should be executed per row by means of the df.apply(lambda row : ..) function.
     
    Parameters
    ----------
    index: int
        the track's index
    track_geometry: LineString
        The track's geometry
    unionStreet: MultiLineString
        The unary union of the street network GeoDataFrame
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    
    Returns
    -------
    tuple
    """
    
    p_origin = Point(track_geometry.coords[0])
    counter = 1
    while (not p_origin.within(unionStreet)): 
        p_origin = Point(track_geometry.coords[counter])
        counter += 1
        
    x_origin, y_origin = track_geometry.coords[counter][0], track_geometry.coords[counter][1] 
    try: origin = nodes_gdf[(nodes_gdf.x == x_origin) & (nodes_gdf.y == y_origin)]['nodeID'].iloc[0].nodeID
    except: origin = ci.distance_geometry_gdf(Point(x_origin, y_origin), nodes_gdf)[1]

    p_destination = Point(track_geometry.coords[-1][0], track_geometry.coords[-1][1])
    counter = -2
    while (not p_origin.within(unionStreet)): 
        p_destination = Point(track_geometry.coords[counter])
        counter += -1

    x_destination, y_destination = track_geometry.coords[counter][0], track_geometry.coords[counter][1] 
    try: destination = nodes[(nodes_gdf.x == x_destination) & (nodes_gdf.y == y_destination)]['nodeID'].iloc[0].nodeID
    except: destination = ci.distance_geometry_gdf(Point(x_destination, y_destination), nodes_gdf)[1]
    
    return (origin, destination)

def line_at_centroid(line_geometry, offset):
    """
    Given a LineString, it creates a LineString that intersects the given geometry at its centroid.
    The offset determines the distance from the original line.
    This fictional line can be used to count precisely the number of trajectories intersecting a segment.
    This function should be executed per row by means of the df.apply(lambda row : ..) function. 
    
    Parameters
    ----------
    line_geometry: LineString
        A street segment geometry
    offset: float
        The offset from the geometry
    
    Returns
    -------
    LineString
    """
    
    left = line_geometry.parallel_offset(offset, 'left')
    right =  line_geometry.parallel_offset(offset, 'right')
    
    if left.geom_type == 'MultiLineString': left = _merge_disconnected_lines(left)
    if right.geom_type == 'MultiLineString': right = merge_disconnected_lines(right)   
    
    if (left.is_empty == True) & (right.is_empty == False): left = line_geometry
    if (right.is_empty == True) & (left.is_empty == False): right = line_geometry
    left_centroid = left.interpolate(0.5, normalized = True)
    right_centroid = right.interpolate(0.5, normalized = True)
   
    fict = LineString([left_centroid, right_centroid])
    return(fict)
    
def count_at_centroid(line_geometry, trajectories_gdf):
    """
    Given a LineString geometry, it counts all the geometries in a LineString GeoDataFrame (the GeoDataFrame containing GPS trajectories).
    This function should be executed per row by means of the df.apply(lambda row : ..) function.
        
    Parameters
    ----------
    line_geometry: LineString
        A street segment geometry
    tracks_gdf: LineString GeoDataFrame
        A set of GPS tracks 
    
    Returns
    -------
    int
    """
    
    intersecting_tracks = trajectories_gdf[trajectories_gdf.geometry.intersects(line_geometry)]
    return len(intersecting_tracks)
    
def traversed_nodes(track_geometry, lines_at_centroids_gdf):
    """
    Given a GPS trajectory geometry and a GeoDataFrame of all the lines built at the street segments' centroid (see above and juptyer notebook), the function generates the list of nodes of a street network traversed by 
    the given route. This function should be executed per row by means of the df.apply(lambda row : ..) function.    
    
    Parameters
    ----------
    track_geometry: LineString
        The track's geometry
    lines_at_centroids_gdf: LineString GeoDataFrame
        A GeoDataFrame of LineString built perpendiculary to the centroid of each street network's segment

    Returns
    -------
    List
    """
    
    tmp = lines_at_centroids_gdf[lines_at_centroids_gdf.geometry.intersects(track_geometry)]
    edgeIDs = tmp['edgeID']
    u = list(lines_at_centroids_gdf[lines_at_centroids_gdf.edgeID.isin(edgeIDs)].u)
    v = list(lines_at_centroids_gdf[lines_at_centroids_gdf.edgeID.isin(edgeIDs)].v)
    traversed_nodes = list(set(u+v))
    try: traversed_nodes.remove(origin)
    except: pass
    try: traversed_nodes.remove(destination)
    except: pass
    
    return traversed_nodes

    
def _merge_disconnected_lines(list_lines):
    """
    Given a list of LineStrings, even disconnected, it merges them in one LineString.
    
    Parameters
    ----------
    list_lines: List of LineString
        A list of LineString to connect
    
    Returns
    -------
    LineString
    """
    
    new_line = []
    for n, i in enumerate(list_lines):
        coords = list(i.coords)
        if n < len(list_lines)-1: coords.append(list_lines[n+1].coords[-1])
        new_line = new_line + coords

    line_geometry = LineString([coor for coor in new_line])
    return(line_geometry)
    
def generate_routes_stats(routes_gdf_list, route_choice_models, labels, titles = None):
    """
    This function generate a translated table that can be used for statistical tests (e.g. Anova, Games-Howell test), for each passed route choice model.
     
    Parameters
    ----------
    routes_gdf_list: List of GeoDataFrames
        A list contianing a GeoDataFrame of routes per each route choice model
    route_choice_models: List of String
        The list of the abbreviation of the route choice models for which the statistics are desired
    labels: List of String
        The labels of the variables on which the statistics should be computed.
    titles: List of String
        The titles of the variables investigated (for visualisation purposes).
    
    Returns
    -------
    Pandas DataFrame
    """
    
    if titles == None: titles = labels
    length = len(routes_gdf_list[0])
    route_choice_models_col = []
    values = []
    type_stats = []

    for n, label in enumerate(labels):
        for nn, model in enumerate(route_choice_models):
            route_choice_models_col = route_choice_models_col + [model] * length
            type_stats = type_stats+ [titles[n]] * length
            col = list(routes_gdf_list[nn][label])
            values = values + col
    
    route_stats = pd.DataFrame({'routeChoice': route_choice_models_col, 'values': values, 'type': type_stats}) 
    return route_stats    
    

def local_landmarkness_track(traversed_nodes, nodes_gdf, buildings_gdf):
    """
    It computes the accumulated local landmarkness of a route (a sequence of traversed nodes). This is designed for computing the local landmarkness of a GPS trajectory.
    This function should be executed per row by means of the df.apply(lambda row : ..) function. 
    
    Parameters
    ----------
    traversed_nodes: List of Node
        The list of traversed nodes by a track
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    
    Returns
    -------
    float
    """

    lL = 0.0
    for n in traversed_nodes:
        if len(nodes_gdf.loc[n].loc_land) == 0: continue
        node_lL = buildings_gdf[buildings_gdf.buildingID.isin(nodes_gdf.loc[n].loc_land)]['lScore_sc'].max()
        lL += node_lL
                
    return lL


def global_landmarkness_track(destination, traversed_nodes, nodes_gdf, buildings_gdf):
    """
    It computes the accumulated global landmarkness of a route (a sequence of traversed nodes). This is designed for computing the global landmarkness of a GPS trajectory.
    This function should be executed per row by means of the df.apply(lambda row : ..) function. 
    
    Parameters
    ----------
    destination: int
        The nodeID of the destination node of a GPS track
    traversed_nodes: List of Node
        The list of traversed nodes by a track
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    
    Returns
    -------
    float
    """
    
    destination_node = nodes_gdf.loc[destination]
    destination_geometry = destination_node.geometry
    anchors = destination_node.anchors
    if len(anchors) == 0: return 0.0
    
    else:
        gL = 0.0
        
        for n in traversed_nodes:
            node = nodes_gdf.loc[n]
            if node.geometry.distance(destination_geometry) <= 300:
                continue
            visible_anchors = [item for item in node.dist_land if item in anchors]
            if len(visible_anchors) == 0: 
                continue
            
            else:
                node_geometry = node.geometry
                distance_destination = destination_geometry.distance(node_geometry)
                node_gL = 0.0
                
                for building in visible_anchors:
                    landmark = buildings_gdf.loc[building]
                    score = landmark.gScore_sc;
                    distance_landmark = destination_geometry.distance(landmark.geometry)
                    if (distance_landmark == 0.0):
                        distance_landmark = 0.1
                    distance_weight = distance_destination/distance_landmark
                    if (distance_weight > 1.0): 
                        distance_weight = 1.0
                    score = score*distance_weight;
                    if score > node_gL: 
                        node_gL = score
            
            gL += node_gL
                         
        return gL

## Landmarkness on routes

def local_landmarkness_route(row, nodes_gdf, edges_gdf, buildings_gdf):
    """
    It computes the accumulated local landmarkness of a route generated by an ABM.
    This function should be executed per row by means of the df.apply(lambda row : ..) function. 
    
    Parameters
    ----------
    row: Pandas Series
        A row of the GeoDataFrame containing the routes
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    edges_gdf: LineString GeoDataFrame
        The GeoDataFrame of the edges of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    
    Returns
    -------
    float
    """
    
    origin = row.O
    destination = row.D

    u = list(edges_gdf[edges_gdf.edgeID.isin(row.edgeIDs)].u)
    v = list(edges_gdf[edges_gdf.edgeID.isin(row.edgeIDs)].v)
    traversed_nodes = list(set(u+v))
    if origin in traversed_nodes: traversed_nodes.remove(origin)
    if destination in traversed_nodes: traversed_nodes.remove(destination)
    
    lL = 0.0
    for n in traversed_nodes:
        if len(nodes_gdf.loc[n].loc_land) == 0: continue
        node_lL = buildings_gdf[buildings_gdf.buildingID.isin(nodes_gdf.loc[n].loc_land)]['lScore_sc'].max()
        lL += node_lL
                
    return lL

    
def global_landmarkness_route(row, nodes_gdf, edges_gdf, buildings_gdf):
    """
    It computes the accumulated global landmarkness of a route generated by an ABM.
    This function should be executed per row by means of the df.apply(lambda row : ..) function.

    Parameters
    ----------
    row: Pandas Series
        A row of the GeoDataFrame containing the routes
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    edges_gdf: LineString GeoDataFrame
        The GeoDataFrame of the edges of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    
    Returns
    -------
    float
    """
    
    origin = row.O
    destination = row.D
    destination_node = nodes_gdf.loc[destination]
    destination_geometry = destination_node.geometry
    anchors = destination_node.anchors
    if len(anchors) == 0: return 0.0
    else:
        gL = 0.0

        u = list(edges_gdf[edges_gdf.edgeID.isin(row.edgeIDs)].u)
        v = list(edges_gdf[edges_gdf.edgeID.isin(row.edgeIDs)].v)
        traversed_nodes = list(set(u+v))
        if origin in traversed_nodes: traversed_nodes.remove(origin)
        if destination in traversed_nodes: traversed_nodes.remove(destination)
        
        for n in traversed_nodes:
            node = nodes_gdf.loc[n]
            if node.geometry.distance(destination_geometry) <= 300:
                continue
            visible_anchors = [item for item in node.dist_land if item in anchors]
            if len(visible_anchors) == 0: 
                continue
            else:
                node_geometry = node.geometry
                distance_destination = destination_geometry.distance(node_geometry)
                node_gL = 0.0
                
                for building in visible_anchors:
                    landmark = buildings_gdf.loc[building]
                    score = landmark.gScore_sc;
                    distance_landmark = destination_geometry.distance(landmark.geometry)
                    if (distance_landmark == 0.0):
                        distance_landmark = 0.1
                    distance_weight = distance_destination/distance_landmark
                    if (distance_weight > 1.0): 
                        distance_weight = 1.0;
                    score = score*distance_weight;
                    if score > node_gL: 
                        node_gL = score
            
            gL += node_gL
                
        return gL
               
## Landmark integration   
    
def assign_anchors_to_nodes(nodes_gdf, buildings_gdf, radius = 2000, threshold = 0.3, nr_anchors = 5):
    """
    The function assigns a set of anchoring or orienting landmark (within a certan radius) to each node in a nodes GeoDataFrame.
    Amongst the buildings GeoDataFrame, only the one with a global landmark score higher than a certain threshold are considered.
    The global landmarks around a node, within the radius, may work as orienting landmarks towards the node, if the landmark is visible in other locations across the city.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    radius: float
        It determintes the area within which the point of references for a node are searched
    threshold: float
        It regulates the selection of global_landmarks buildings (lanmdarks), by filtering out buildings whose global landmark score is lower than the argument.
    nr_anchors: int
        The maximum number of anchors per node. This speeds up further computations and also takes into account the fact that people may associate only a certain
        ammount of anchors with a destination (the anchors selected are the always the best in terms of score; e.g. if nr_anchors == 5, the best 5 anchors in terms of gScore_sc are kept).
        
    Returns
    -------
    GeoDataFrame
    """

    global_landmarks = buildings_gdf[buildings_gdf.gScore_sc >= threshold]
    global_landmarks = global_landmarks.round({"gScore_sc":3})
    sindex = global_landmarks.sindex

    nodes_gdf["anchors"] = nodes_gdf.apply(lambda row: _find_anchors(row["geometry"], global_landmarks, sindex, radius, nr_anchors), axis = 1)

    return nodes_gdf    
    
def _find_anchors(node_geometry, global_landmarks, global_landmarks_sindex, radius, nr_anchors):
    """
    The function finds the set of anchoring or orienting landmark (within a certan radius) to a given node.
     
    Parameters
    ----------
    node_geometry: Point
        The geometry of the node considered
    global_landmarks: Polygon GeoDataFrame
        The GeoDataFrame of the global_landmarks buildings of a city
    global_landmarks_sindex: Rtree spatial index
        The spatial index on the GeoDataFrame of the global landmarks of a city
    radius: float
        It determintes the area within which the point of references for a node are searched
    nr_anchors: int
        The maximum number of anchors per node. This speeds up further computations and also takes into account the fact that people may associate only a certain
        ammount of anchors with a destination (the anchors selected are the always the best in terms of score; e.g. if nr_anchors == 5, the best 5 anchors in terms of gScore_sc are kept).
    
    Returns
    -------
    List
    """

    list_anchors = []
    b = node_geometry.buffer(radius) 
    possible_matches_index = list(global_landmarks_sindex.intersection(b.bounds))
    possible_matches = global_landmarks.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(b)]
    
    if len(precise_matches) == 0: 
        pass
    else:  
        precise_matches.sort_values(by = "gScore_sc", ascending = False, inplace = True)
        anchors = precise_matches.iloc[0:nr_anchors]
        list_anchors = anchors["buildingID"].tolist()
   
    return list_anchors


def assign_3d_visible_landmarks_to_nodes(nodes_gdf, buildings_gdf, sight_lines, threshold = 0.3):
    """
    The function assigns to each node in a nodes' GeoDataFrame the set of visibile buildings, on the basis of pre-computed 3d sight_lines.
    Only global landmarks, namely buildings with global landmarkness higher than a certain threshold, are considered as buildings.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    sight_lines: LineString GeoDataFrame
        The GeoDataFrame of 3d sight lines from nodes to buildings. 
        The nodeID and buildingID fields are expected to be in this GeoDataFrame, referring respectively to obeserver and target of the line
    threshold: float
        It regulates the selection of global_landmarks buildings (lanmdarks), by filtering out buildings whose global landmark score is lower than the argument
   
    Returns
    -------
    GeoDataFrame
    """
    
    global_landmarks = buildings_gdf[buildings_gdf.gScore_sc >= threshold]
    global_landmarks = global_landmarks.round({"gScore_sc":3})
    index_global_landmarks = buildings_gdf["buildingID"].values.astype(int) 
    sight_lines_to_global_landmarks = sight_lines[sight_lines["buildingID"].isin(index_global_landmarks)]

    nodes_gdf["dist_land"] = nodes_gdf.apply(lambda row: _find_visible_landmarks(row["nodeID"], global_landmarks, sight_lines_to_global_landmarks), axis = 1)
    return nodes_gdf
    
def _find_visible_landmarks(nodeID, global_landmarks, sight_lines_to_global_landmarks):
    """
    The function finds the set of visibile buildings from a certain node, on the basis of pre-computed 3d sight_lines.
    Only global landmarks, namely buildings with global landmarkness higher than threshold, are considered as buildings.
     
    Parameters
    ----------
    nodesID: int
        The nodeID of the node considered
    global_landmarks: Polygon GeoDataFrame
        The GeoDataFrame of buildings considered global landmarks
    sight_lines_to_global_landmarks: LineString GeoDataFrame
        The GeoDataFrame of 3d sight lines from nodes to buildings (only landmarks).
        The nodeID and buildingID fields are expected to be in this GeoDataFrame, referring respectively to obeserver and target of the line.
        
    Returns
    -------
    List
    """

    # per each node, the sight lines to the global_landmarks landmarks are extracted.  
    global_landmarks_list, scores_list = [], []
    sight_node = sight_lines_to_global_landmarks[sight_lines_to_global_landmarks["nodeID"] == nodeID] 
    ix_global_landmarks_node = list(sight_node["buildingID"].values.astype(int))
    global_landmarks_from_node = global_landmarks[global_landmarks["buildingID"].isin(ix_global_landmarks_node)] 
    global_landmarks_from_node.sort_values(by = "gScore_sc", ascending = False, inplace = True)
    if len(global_landmarks_from_node) == 0: 
        pass
    else:
        global_landmarks_list = global_landmarks_from_node["buildingID"].tolist()         

    return global_landmarks_list
    
def assign_local_landmarks_to_nodes(nodes_gdf, buildings_gdf, radius = 50):
    """
    The function assigns a set of adjacent buildings (within a certain radius) to each node in a nodes GeoDataFrame.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network   
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    radius: float
        The radius which regulates the search of adjacent buildings
        
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf = buildings_gdf.round({"gScore_sc":3})
    sindex = buildings_gdf.sindex
    nodes_gdf["loc_land"] = nodes_gdf.apply(lambda row: _find_local_landmarks(row["geometry"], buildings_gdf, sindex, radius), axis = 1)
    return nodes_gdf
    
def _find_local_landmarks(node_geometry, buildings_gdf, buildings_gdf_sindex, radius):
    """
    The function finds the set of adjacent buildings (within a certain radius) for a given node.
     
    Parameters
    ----------
    node_geometry: Point
        The geometry of the node considered
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    buildings_gdf_sindex: Rtree spatial index
        The spatial index on the GeoDataFrame of the buildings of a city
    radius: float
        The radius which regulates the search of adjacent buildings
        
    Returns
    -------
    List
    """
    
    list_local = []
    b = node_geometry.buffer(radius)    
    possible_matches_index = list(buildings_gdf_sindex.intersection(b.bounds))
    possible_matches = buildings_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(b)]
    
    if len(precise_matches) == 0: 
        pass
    else:
        precise_matches = precise_matches.sort_values(by = "lScore_sc", ascending = False).reset_index()
        list_local = precise_matches["buildingID"].tolist()
        list_scores = precise_matches["lScore_sc"].tolist()
    
    return list_local
    
    
    
    
def regionBased_route_stats(edgeIDs, route_geometry, nodes_gdf, edges_gdf):
    """
    This function generate a translated table that can be used for statistical tests (e.g. Anova, Games-Howell test), for each passed route choice model.
     
    Parameters
    ----------
    routes_gdf_list: List of GeoDataFrames
        A list contianing a GeoDataFrame of routes per each route choice model
    route_choice_models: List of String
        The list of the abbreviation of the route choice models for which the statistics are desired
    labels: List of String
        The labels of the variables on which the statistics should be computed.
    titles: List of String
        The titles of the variables investigated (for visualisation purposes).
    
    Returns
    -------
    Pandas DataFrame
    """    
    
    districts = {}
    for edgeID in edgeIDs:
        edgeID = int(edgeID)
        u = edges_gdf.loc[edgeID].u
        v = edges_gdf.loc[edgeID].v
        length = edges_gdf.loc[edgeID].geometry.length
        if nodes_gdf.loc[u].district == nodes_gdf.loc[v].district:
            d = nodes_gdf.loc[u].district 
            districts[d] = round(districts.get(d, 0) + length, 2)
    
    pedestrian_length = edges_gdf[(edges_gdf.edgeID.isin(edgeIDs)) & (edges_gdf['pedestrian'] == 1)]['length'].sum()/route_geometry.length
    major_roads_length = edges_gdf[(edges_gdf.edgeID.isin(edgeIDs)) & (edges_gdf['highway'] == 'primary')]['length'].sum()/route_geometry.length
    p_barrier_length = edges_gdf[(edges_gdf.edgeID.isin(edgeIDs)) & (edges_gdf['p_bool'] == 1)]['length'].sum()/route_geometry.length  
    return pedestrian_length, major_roads_length, p_barrier_length, districts
    
def portion_region(row, nodes, which = 'first'):


def count_regions(row, nodes):
    count = 0
    if not nodes.loc[int(row['O'])].district in row['districts']:
        count += 1
    if not nodes.loc[int(row['D'])].district in row['districts']:
        count +=1
    return len(row['districts']) + count
    
 def generate_ax_hcolorbar(cmap, fig, ax, nrows, ncols, text_color, font_size, norm = None, ticks = 5, symbol = False):
    
    if font_size is None: 
        font_size = 20
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    vr_p = 1/30.30
    hr_p = 0.5/30.30
    
    width = ax.get_position().width
    x = ax.get_position().x0
    y = ax.get_position().y0 - 0.070
    height = 0.025
    pos = [x, y, width, height]
    cax = fig.add_axes(pos, frameon = False)
    cax.tick_params(size=0)
    cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
    tick_locator = ticker.MaxNLocator(nbins=ticks)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.outline.set_visible(False)
     
    if symbol: cax.set_xticklabels([round(t,1) if t < norm.vmax else "> "+str(round(t,1)) for t in cax.get_xticks()])
    else: cax.set_xticklabels([round(t,1) for t in cax.get_xticks()])
    
    plt.setp(plt.getp(cax.axes, "xticklabels"), size = 0, color = text_color, fontfamily = 'Times New Roman', 
             fontsize=(font_size-font_size*0.33))