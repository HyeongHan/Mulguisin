import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union, polygonize


default_cosmo = FlatLambdaCDM(
    H0=67.66,      
    Om0=0.3111,    
    Tcmb0=2.7255,  
    Ob0=0.049      
)


def alpha_shape(points, alpha=5):
    """
    Compute the alpha shape (concave hull) of a set of points.
    alpha: larger value → more detailed boundary 

    # Example input
    points = np.column_stack((x, y))
    shape = alpha_shape(points, alpha=5)

    plt.figure(figsize=(6,6))
    plt.scatter(points[:,0], points[:,1], s=5)
    xs, ys = shape.exterior.xy
    plt.plot(xs, ys, 'r-', linewidth=2)

    plt.title("α-shape / Concave Hull Boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    """
    if len(points) < 4:
        return MultiPoint(points).convex_hull

    tri = Delaunay(points)
    edges = []

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]

        # Compute lengths of sides
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)

        # Compute triangle area
        s = (a + b + c) / 2.0
        area = max(s * (s - a) * (s - b) * (s - c), 0)**0.5

        if area == 0:
            continue

        # Radius of circumcircle
        R = a*b*c / (4.0 * area)

        # Keep triangles with small circumradius
        if R < 1.0 / alpha:
            edges.append((ia, ib))
            edges.append((ib, ic))
            edges.append((ic, ia))

    # Get unique edges
    edge_set = set(tuple(sorted(e)) for e in edges)

    # Build polygon
    m = MultiPoint(points)
    polys = polygonize([
        [points[e[0]], points[e[1]]] for e in edge_set
    ])
    #return unary_union(list(polys))
    EPS = 100
    single_polygon = unary_union([p.buffer(EPS) for p in polys]).buffer(-EPS)
    return single_polygon # this forcefully make multipolygon into polygon



def tan_project(ra, dec, ra0, dec0):
    ra  = np.radians(ra)
    dec = np.radians(dec)
    ra0 = np.radians(ra0)
    dec0 = np.radians(dec0)

    cosc = np.sin(dec0)*np.sin(dec) + np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0)
    
    x =  np.cos(dec) * np.sin(ra - ra0) / cosc      # radians
    y = (np.cos(dec0)*np.sin(dec) - np.sin(dec0)*np.cos(dec)*np.cos(ra - ra0)) / cosc

    return x, y

def sky_to_comoving_xy(ra, dec, ra0, dec0, z, cosmology=cosmo):
    """
    RA, Dec, RA0, Dec0 in degrees, z is the redshift at which you want comoving coords.
    Returns X, Y in comoving Mpc relative to (ra0, dec0).
    """
    # Step 1: angular TAN projection (radians)
    x_rad, y_rad = tan_project(ra, dec, ra0, dec0)

    # Step 2: transverse comoving distance at redshift z (Mpc)
    D_M = cosmology.comoving_transverse_distance(z).value  # Mpc

    # Step 3: convert to comoving Mpc
    X = D_M * x_rad
    Y = D_M * y_rad

    return X, Y
