"""Lane-level map for a nuScenes scene: OSM roads vs nuScenes surveyed lanes vs our COLMAP reconstruction.
Built up one method at a time."""
import numpy as np

# Geographic position of each nuScenes map's origin (x = 0, y = 0), its south-west corner.
# From nuscenes-devkit scripts/export_poses.py. Map x points east, y points north, in metres.
MAP_ORIGIN = {
    "boston-seaport": (42.336849169438615, -71.05785369873047),
    "singapore-onenorth": (1.2882100868743724, 103.78475189208984),
    "singapore-hollandvillage": (1.2993652317780957, 103.78217697143555),
    "singapore-queenstown": (1.2782562240223188, 103.76741409301758),
}
EARTH_RADIUS = 6378137.0  # metres, WGS84 equatorial radius (same value the devkit uses)


def map_to_latlon(location, x, y):
    """nuScenes map metres (x east, y north) -> latitude, longitude in degrees. Works on arrays.

    Same method as the devkit: treat (x, y) as "walk `distance` metres from the origin at compass
    `bearing`" and find where you land on a sphere (the great-circle destination formula). A simpler
    flat approximation, lat += y/R and lon += x/(R cos lat0), drifts by up to ~0.8 m across the
    3 km map because a metre east spans more longitude the further north you are."""
    lat0, lon0 = np.radians(MAP_ORIGIN[location])
    x, y = np.asarray(x, float), np.asarray(y, float)
    bearing = np.arctan2(x, y)  # clockwise from north
    d = np.hypot(x, y) / EARTH_RADIUS  # distance as an angle at the Earth's centre
    lat = np.arcsin(np.sin(lat0) * np.cos(d) + np.cos(lat0) * np.sin(d) * np.cos(bearing))
    lon = lon0 + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(lat0), np.cos(d) - np.sin(lat0) * np.sin(lat))
    return np.degrees(lat), np.degrees(lon)


def latlon_to_map(location, lat, lon):
    """Latitude, longitude in degrees -> nuScenes map metres (x east, y north). Works on arrays.

    Exact inverse of map_to_latlon: find the great-circle distance and starting bearing from the map
    origin to the point, then split that walk into its east (x) and north (y) parts."""
    lat0, lon0 = np.radians(MAP_ORIGIN[location])
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    dlon = lon - lon0
    # haversine: angle at the Earth's centre between origin and point (stable for short distances)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2) ** 2
    d = 2 * np.arcsin(np.sqrt(a)) * EARTH_RADIUS
    bearing = np.arctan2(np.sin(dlon) * np.cos(lat),
                         np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon))
    return d * np.sin(bearing), d * np.cos(bearing)
