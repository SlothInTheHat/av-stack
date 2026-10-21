"""Download OpenStreetMap roads for a bounding box and convert them to OpenDRIVE with CARLA's built-in converter.
Usage: python osm_to_carla.py NAME SOUTH WEST NORTH EAST [--load]
Writes mapping/maps/NAME.osm and NAME.xodr; --load also generates the world in a running CARLA server."""
import argparse
import urllib.parse
import urllib.request
from pathlib import Path

import carla

OVERPASS = "https://overpass-api.de/api/interpreter"
HIGHWAYS = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]


def fetch_osm(s, w, n, e):
    q = f'[out:xml][timeout:90];(way["highway"]({s},{w},{n},{e});>;);out;'
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "av-stack-capstone/0.1 (student project)"})
    return urllib.request.urlopen(req, timeout=120).read().decode()


def to_xodr(osm):
    st = carla.Osm2OdrSettings()
    st.set_osm_way_types(HIGHWAYS)
    st.default_lane_width = 3.5
    st.center_map = True  # CARLA world origin = middle of the bbox
    return carla.Osm2Odr.convert(osm, st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("south", type=float)
    ap.add_argument("west", type=float)
    ap.add_argument("north", type=float)
    ap.add_argument("east", type=float)
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    out = Path(__file__).parent / "maps"
    osm = fetch_osm(a.south, a.west, a.north, a.east)
    (out / f"{a.name}.osm").write_text(osm, encoding="utf-8")
    print(f"OSM: {osm.count('<way ')} ways, {osm.count('<node ')} nodes")
    xodr = to_xodr(osm)
    (out / f"{a.name}.xodr").write_text(xodr, encoding="utf-8")
    print(f"OpenDRIVE: {xodr.count('<road ')} roads, {xodr.count('<junction ')} junctions -> {out / (a.name + '.xodr')}")
    if a.load:
        client = carla.Client(a.host, 2000)
        client.set_timeout(120)
        client.generate_opendrive_world(xodr, carla.OpendriveGenerationParameters(
            vertex_distance=2.0, max_road_length=50.0, wall_height=0.0, additional_width=0.6,
            smooth_junctions=True, enable_mesh_visibility=True))
        print("world generated; spawn points:", len(client.get_world().get_map().get_spawn_points()))


if __name__ == "__main__":
    main()
