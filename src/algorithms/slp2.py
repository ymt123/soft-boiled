from math import floor, radians, sin, cos, asin, sqrt, pi
import numpy as np
import itertools
from collections import namedtuple

GeoCoord = namedtuple('GeoCoord', ['lat', 'lon'])
Vertex = namedtuple('Vertex', ['id', 'geo_coord'])

def dispersion(distance_func, geo_source, geo_neighbors):
    distances = map(lambda geo_neighbor: distance_func(geo_source,geo_neighbor), geo_neighbors)
    arr = np.array(distances)
    return np.median(arr)

EARTH_RADIUS = 6367
def haversine(x, y):
    """
        From: http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
    """

    #if either distance is None return None
    if x is None or y is None:
        raise Exception("Null coordinate")

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [x.lon, x.lat, y.lon, y.lat])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(min(1, sqrt(a))) # Added min to protect against roundoff errors for nearly antipodal locations
    return c * EARTH_RADIUS

def median(distance_func, vertices, weights=None):
    """
        given a python list of vertices, and a distance function, this will find the vertex that is most central
        relative to all other vertices. All of the vertices must have geocoords
    """

    #get the distance between any two points
    distances = map(lambda (v0, v1) :distance_func(v0.geo_coord, v1.geo_coord), itertools.combinations(vertices, 2))

    #create a dictionary with keys representing the index of a location
    m = { a: list() for a in range(len(vertices)) }

    #add the distances from each point to the dict
    for (k0,k1),distance in zip(itertools.combinations(range(len(vertices)), 2) , distances):
        #a distance can be None if one of the vertices does not have a geocoord
        if(weights is None):
            m[k0].append(distance)
            m[k1].append(distance)
        else:
            m[k0].append(distance/weights[k0])
            m[k1].append(distance/weights[k1])

    summed_values = map(sum, m.itervalues())

    return vertices[summed_values.index(min(summed_values))]


def get_known_locs(table_name, min_locs=3):
    '''
        Given a loaded twitter table, this will return all the twitter users with locations. A user's location is determined
        by the median location of all known tweets. A user must have at least min_locs locations in order for a location to be
        estimated
    '''
    return sqlCtx.sql('select user.id_str, geo.coordinates from %s where geo.coordinates is not null' % table_name)\
        .map(lambda row: (row.id_str, row.coordinates)).groupByKey()\
        .filter(lambda (id_str,coord_list): len(coord_list) > 3)\
            .map(lambda (id_str,coords): (id_str, median(haversine, [Vertex(id_str, GeoCoord(lat,lon)) for lat,lon in coords]))).cache()


def get_edge_list(table_name):
    '''
        Given a loaded twitter table, this will return the @mention network in the form (src_id, (dest_id, num_@mentions))
    '''
    tmp_edges = sqlCtx.sql('select user.id_str, entities.user_mentions from %s where size(entities.user_mentions) > 0' %\
                           slp.options['temp_table_name']).flatMap(lambda row : [((row.id_str, mentioned_user.id_str),1)\
                                                                                 for mentioned_user in row.user_mentions]).reduceByKey(lambda x,y:x+y)
        
    return tmp_edges.map(lambda ((src_id,dest_id),num_mentions):\
                                      ((dest_id,src_id),num_mentions)).join(tmp_edges).\
            map(lambda ((src_id,dest_id), (count0, count1)): (src_id, (dest_id, min(count0,count1)))).cache()



def run(table_name):
    
    locs_known = get_known_locs(table_name)
    edge_list = get_edge_list(slp.options(table_name)
    result = train(locs_known, edge_list)


def train(locs_known, edge_list, num_iters, dispersion_threshold=50):
'''
  Inputs:
  locs_known => (src_id, vertex)
  edge_list  => (src_id, (dest_id, weight))
  line 0:  attach the locations to each of the sources in the edges... (src_id, ((dst_id, weight), src_vertex))
  line 1:  get the src and coord in value in prep for group by...      (dst_id, (Vertex, weight))
  line 2:  filter out those edges where a Vertex has no geoCoord...    (dst_id, (Vertex, weight)) #has geocoord
  line 3:  group by in prep for neighbor operations...                 (dst_id, [(Vertex, weight)..])
  line 4:  filter out nodes with fewer than 2 neighbors...             (dst_id, [(Vertex, weight)..]) # >2
  line 5:  add back in known locs so we only predict unknown...        (dst_id, ([(Vertex, weight)..], hasLoc))
  line 6:  only keep the nodes we are trying to predict...             (dst_id, ([(Vertex, weight)..], hasLoc))
  line 7:  apply the median to the neighbors...                        (dst_id, (median_vtx, neighbors))
  line 8:  given the median, filter out high dispersion....            (dst_id, (median_vtx, neighbors)) <disp
  line 9:  prepare for the union by adjusting format...                (dst_id, median_coord)
  line 8:  union to create the global location rdd...                  (dst_id, median_geoCoord)
  '''
    NEIGHBOR_THRESHOLD = 2

    l = locs_known

    for i in range(num_iters):
        l = edge_list.leftOuterJoin(l)\
        .map(lambda (src_id, ((dst_id, weight), known_vertex)) : (dst_id, (known_vertex, weight)))\
        .filter(lambda (dst_id, (known_vertex, weight) ): known_vertex != None)\
        .groupByKey()\
        .filter(lambda (src_id, neighbors) : neighbors.maxindex > NEIGHBOR_THRESHOLD)\
        .leftOuterJoin(locs_known)\
        .filter(lambda (src_id, (neighbors, hasLoc)) : hasLoc is None)\
        .map(lambda (src_id, (neighbors, locLoc)) :\
           (src_id, (median(haversine, [v for v,w in neighbors],[w for v,w in neighbors]), neighbors)))\
        .filter(lambda (src_id, (median_vertex, neighbors)) :\
              dispersion(haversine, median_vertex.geo_coord,\
                         [vtx.geo_coord for (vtx,weight) in neighbors]) < dispersion_threshold)\
        .flatMapValues(lambda (src_id, (median_vertex, neighbors)) : median_vertex.geo_coord)\
        .union(locs_known)

    return l





