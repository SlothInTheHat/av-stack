"""Phase 1: drive a loop of a CARLA town with pure pursuit (steering) + PID (speed)."""
import argparse
import math
import subprocess
import time

import carla
import numpy as np

from avstack_control import PID, PurePursuit


def wsl_host():
    # WSL2 NAT: the Windows host is the default gateway
    out = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True).stdout
    return out.split()[2]


def build_path(world, spacing=1.0, length=600):
    m = world.get_map()
    spawns = m.get_spawn_points()
    if spawns:
        wp = spawns[0]
        start = m.get_waypoint(wp.location)
    else:  # generated OpenDRIVE worlds may have no spawn points
        start = m.generate_waypoints(2.0)[0]
        wp = start.transform
        wp.location.z += 0.5
    path, w = [start], start
    for _ in range(int(length / spacing)):
        w = w.next(spacing)[0]
        path.append(w)
    return wp, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--xodr", default=None, help="OpenDRIVE file to generate the world from instead of --town")
    ap.add_argument("--speed", type=float, default=8.0, help="target m/s")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--weather", default=None, help="carla.WeatherParameters preset name, or golden")
    ap.add_argument("--realtime", action="store_true", help="pace ticks to wall clock (for recording)")
    ap.add_argument("--camera", choices=["top", "chase"], default="top")
    a = ap.parse_args()

    client = carla.Client(a.host or wsl_host(), a.port)
    client.set_timeout(30)
    if a.xodr:
        client.generate_opendrive_world(open(a.xodr, encoding="utf-8").read(), carla.OpendriveGenerationParameters(
            vertex_distance=2.0, max_road_length=50.0, wall_height=0.0, additional_width=0.6,
            smooth_junctions=True, enable_mesh_visibility=True))
        world = client.get_world()
    else:
        world = client.load_world(a.town)
    if a.weather == "golden":
        world.set_weather(carla.WeatherParameters(
            cloudiness=25, precipitation=0, sun_altitude_angle=18, sun_azimuth_angle=250,
            fog_density=3, wetness=0, scattering_intensity=1.0))
    elif a.weather:
        world.set_weather(getattr(carla.WeatherParameters, a.weather))
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    world.apply_settings(settings)
    dt = settings.fixed_delta_seconds

    spawn, wps = build_path(world)
    path = [(w.transform.location.x, w.transform.location.y) for w in wps]
    for w in wps:  # path points: green dots on the lane centre
        world.debug.draw_point(w.transform.location + carla.Location(z=0.3), size=0.08,
                               color=carla.Color(0, 255, 0), life_time=1000)
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    car = world.spawn_actor(bp, spawn)
    pp, pid = PurePursuit(wheelbase=2.9), PID(0.8, 0.05, 0.0)
    cte, last_tgt = [], 0
    try:
        for _ in range(a.steps):
            t0 = time.perf_counter()
            world.tick()
            t, v = car.get_transform(), car.get_velocity()
            speed = math.hypot(v.x, v.y)
            steer, tgt = pp.steer(t.location.x, t.location.y, math.radians(t.rotation.yaw), speed, path)
            if tgt < last_tgt:  # target index wrapped around: end of path reached
                break
            last_tgt = tgt
            u = pid.step(a.speed - speed, dt)
            car.apply_control(carla.VehicleControl(
                steer=steer, throttle=max(u, 0.0), brake=max(-u, 0.0)))
            nearest = world.get_map().get_waypoint(t.location)
            cte.append(t.location.distance(nearest.transform.location))
            # target point (the carrot pure pursuit steers toward): red
            world.debug.draw_point(wps[tgt].transform.location + carla.Location(z=0.6), size=0.2,
                                   color=carla.Color(255, 0, 0), life_time=dt * 2)
            if a.camera == "top":
                cam = carla.Transform(t.location + carla.Location(z=30), carla.Rotation(pitch=-90))
            else:
                back = t.get_forward_vector() * -10
                cam = carla.Transform(t.location + back + carla.Location(z=5),
                                      carla.Rotation(pitch=-15, yaw=t.rotation.yaw))
            world.get_spectator().set_transform(cam)
            if a.realtime:
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    finally:
        car.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    print(f"mean |cross-track to lane centre| = {np.mean(cte):.2f} m, max = {np.max(cte):.2f} m")


if __name__ == "__main__":
    main()
