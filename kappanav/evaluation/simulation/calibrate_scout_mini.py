"""Measure the Scout Mini's plant in Isaac physics: forward-speed response, yaw response by forward speed, and which
steps and slopes it drives over. The results are the SCOUT_MINI dynamics in kappanav.common.physical.

The URDF (AgileX ugv_gazebo_sim + the front Gemini mount) supplies geometry only: its inertials are a Gazebo template
(72 kg; a 3 kg wheel with 0.72 kg m^2), so the model is given the published mass (26 kg) with solid-body inertias of
its own geometry. Motors: four 150 W hub motors; the torque limit is the rated 150 W at the published top speed
(2.7 m/s). Rubber on concrete: friction 1.0, as the URDF's Gazebo contact.

    KAPPANAV_ROBOT=scout_mini bash scripts/evaluation/calibrate_scout_mini.sh --urdf-dir <scout_front_camera> --output out.json
"""
import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

MASS_KG = 26.0                 # published
WHEEL_MASS_KG = 1.5            # each; the body carries the rest
BODY_BOX_M = (.56, .39, .18)   # the chassis mesh's extent (length, width, height) for its inertia
TOP_SPEED_MPS, MOTOR_W = 2.7, 150.
WHEEL_DAMPING = 10.            # velocity-drive gain, N m s / rad: wheel speed tracking limited by the torque cap
PHYSICS_DT, CONTROL_DT = .01, .1
LANE_M = 4.
WHEEL_RADIUS = .080


def scout_urdf(urdf_dir, out_dir, wheel_radius):
    """The package's URDF with absolute mesh paths, the published mass distribution and cylinder tyres (a convex hull
    of the tread mesh rolls as a polygon: 0.894 of the commanded speed at zero torque)."""
    text = (Path(urdf_dir) / 'urdf' / 'scout_front_camera.urdf').read_text()
    text = text.replace('package://scout_front_camera/', str(Path(urdf_dir).resolve()) + '/')
    body = MASS_KG - 4 * WHEEL_MASS_KG
    l, w, h = BODY_BOX_M
    box = dict(ixx=body * (w * w + h * h) / 12, iyy=body * (l * l + h * h) / 12, izz=body * (l * l + w * w) / 12)
    # wheel joints turn the wheel frame so that its local z is the axle (URDF axis 0 0 -1)
    width = .132
    axle, across = WHEEL_MASS_KG * wheel_radius ** 2 / 2, WHEEL_MASS_KG * (3 * wheel_radius ** 2 + width ** 2) / 12
    wheel = dict(ixx=across, iyy=across, izz=axle)

    def inertial(match, mass, inertia):
        block = match.group(0)
        block = re.sub(r'<mass value="[^"]*"', f'<mass value="{mass}"', block)
        return re.sub(r'<inertia [^/]*/>', '<inertia ' + ' '.join(f'{k}="{v:.6g}"' for k, v in inertia.items())
                      + ' ixy="0" ixz="0" iyz="0" />', block)
    links = re.split(r'(?=<link name=)', text)
    for i, link in enumerate(links):
        name = re.match(r'<link name="([^"]+)"', link)
        if not name:
            continue
        if name.group(1) == 'inertial_link':
            links[i] = re.sub(r'<inertial>.*?</inertial>', lambda m: inertial(m, body, box), link, flags=re.S)
        elif name.group(1).endswith('wheel_link'):
            link = re.sub(r'<inertial>.*?</inertial>', lambda m: inertial(m, WHEEL_MASS_KG, wheel), link, flags=re.S)
            # the tread mesh spans z in [-0.047, 0.085] of the wheel frame (axle = z): its centre is 0.019 outboard
            links[i] = re.sub(r'<collision>.*?</collision>', f'<collision><origin xyz="0 0 0.019" rpy="0 0 0" />'
                              f'<geometry><cylinder radius="{wheel_radius}" length="{width}" /></geometry></collision>',
                              link, flags=re.S)
    path = Path(out_dir) / 'scout_mini.urdf'
    path.write_text(''.join(links))
    return path


def convert(urdf, out_dir):
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    cfg = UrdfConverterCfg(asset_path=str(urdf), usd_dir=str(out_dir), usd_file_name='scout_mini.usd', fix_base=False,
                           merge_fixed_joints=True, force_usd_conversion=True,
                           joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='velocity',
                               gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=WHEEL_DAMPING)))
    return Path(UrdfConverter(cfg).usd_path)


def lanes():
    """(name, kind, parameter) per lane; every robot starts at x = 0 heading +x on its own lane."""
    out = [('speed', 'speed', .5)]
    out += [(f'yaw_v{v}', 'yaw', v) for v in (0., .075, .15, .25, .45)]
    out += [(f'up_{h:.2f}', 'step_up', h) for h in (.03, .05, .07, .09, .11)]
    out += [(f'down_{h:.2f}', 'step_down', h) for h in (.03, .05, .07, .09, .11)]
    out += [(f'slope_{a}', 'slope', a) for a in (15, 20, 25, 30, 35)]
    return out


def build(robot_usd, wheel_radius, torque_nm, friction):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    material = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction, friction_combine_mode='max')
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device='cuda:0', physics_material=material))
    sim_utils.GroundPlaneCfg(physics_material=material).func('/World/ground', sim_utils.GroundPlaneCfg(physics_material=material))
    solid = lambda size: sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                                             physics_material=material)
    robots = {}
    for k, (name, kind, p) in enumerate(lanes()):
        y = k * LANE_M
        z0 = .0
        if kind == 'step_up':
            cfg = solid((8., 2., p)); cfg.func(f'/World/obstacle_{k}', cfg, translation=(1.0 + 4., y, p / 2))
        elif kind == 'step_down':
            cfg = solid((2., 2., p)); cfg.func(f'/World/obstacle_{k}', cfg, translation=(0., y, p / 2)); z0 = p
        elif kind == 'slope':
            a = math.radians(p); L, T = 6., .2
            # top face passes through (x = 0.8, z = 0); centre half a thickness below the face
            cx, cz = .8 + L / 2 * math.cos(a) + T / 2 * math.sin(a), L / 2 * math.sin(a) - T / 2 * math.cos(a)
            cfg = solid((L, 2., T))
            cfg.func(f'/World/obstacle_{k}', cfg, translation=(cx, y, cz),
                     orientation=(math.cos(-a / 2), 0., math.sin(-a / 2), 0.))
        robots[name] = Articulation(ArticulationCfg(prim_path=f'/World/robot_{k}',
            spawn=sim_utils.UsdFileCfg(usd_path=str(robot_usd), activate_contact_sensors=False),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., y, z0 + .19)),
            actuators={'wheels': ImplicitActuatorCfg(joint_names_expr=['.*wheel'], stiffness=0., damping=WHEEL_DAMPING,
                effort_limit_sim=torque_nm, velocity_limit_sim=TOP_SPEED_MPS / wheel_radius)}))
    sim.reset()
    return sim, robots


def run(sim, robots, wheel_radius, track):
    import torch
    plan = {name: (kind, p) for name, kind, p in lanes()}
    for r in robots.values():
        r.update(PHYSICS_DT)
    # Joint signs: which wheel velocity rolls the robot forward (left and right axles point opposite ways).
    names = robots['speed'].joint_names
    left = torch.tensor([('left' in n) for n in names], device='cuda:0')
    sign = torch.where(left, -1., 1.)
    settle, seconds = 1., 12.
    log = {name: [] for name in robots}
    steps = int(round(seconds / PHYSICS_DT))
    for i in range(steps):
        t = i * PHYSICS_DT
        for name, robot in robots.items():
            kind, p = plan[name]
            v, w = 0., 0.
            if t >= settle:
                if kind == 'speed':
                    v = p
                elif kind == 'yaw':
                    v, w = p, .5
                else:
                    v = .5
            wheel = torch.where(left, (v - w * track / 2) / wheel_radius, (v + w * track / 2) / wheel_radius) * sign
            robot.set_joint_velocity_target(wheel[None])
            robot.write_data_to_sim()
        sim.step(render=False)
        for name, robot in robots.items():
            robot.update(PHYSICS_DT)
            d = robot.data
            q = d.root_quat_w[0].tolist()
            yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            tilt = math.degrees(math.acos(max(-1., min(1., 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
            vel = d.root_lin_vel_w[0].tolist()
            log[name].append((t, *d.root_pos_w[0].tolist(), yaw, vel[0] * math.cos(yaw) + vel[1] * math.sin(yaw),
                              d.root_ang_vel_w[0, 2].item(), tilt, *(d.joint_vel[0] * sign).tolist(),
                              *d.applied_torque[0].tolist()))
    return {k: np.asarray(v) for k, v in log.items()}, settle


def summarize(log, settle):
    out = {}
    t, x, y, z, yaw, fwd, wz, tilt = log['speed'].T[:8]
    on = t >= settle
    final = np.median(fwd[on & (t >= settle + 3)])
    reach = t[on & (fwd >= (1 - math.exp(-1)) * final)][0] - settle
    steady = log['speed'][on & (t >= settle + 3)]
    out['speed'] = dict(command_mps=.5, steady_mps=round(float(final), 4), response_s=round(float(reach), 3),
                        wheel_rim_mps=[round(float(np.median(steady[:, 8 + j])) * WHEEL_RADIUS, 4) for j in range(4)],
                        wheel_torque_nm=[round(float(np.median(steady[:, 12 + j])), 3) for j in range(4)])
    gains = {}
    for v in (0., .075, .15, .25, .45):
        t, x, y, z, yaw, fwd, wz, tilt = log[f'yaw_v{v}'].T[:8]
        window = t >= settle + 4
        gains[v] = round(float(np.median(wz[window]) / .5), 3)
    out['yaw_gain_by_speed'] = gains
    for kind in ('up', 'down', 'slope'):
        res = {}
        for name, data in log.items():
            if not name.startswith(kind + '_'):
                continue
            t, x, y, z, yaw, fwd, wz, tilt = data.T[:8]
            p = float(name.split('_')[1])
            if kind == 'up':
                ok = x[-1] > 1.0 + .7 and z[-1] - z[0] > p - .01
            elif kind == 'down':
                ok = x[-1] > 1.0 + .7 and z[0] - z[-1] > p - .01 and tilt[-1] < 20
            else:
                ok = z[-1] - z[0] > .5
            res[name] = dict(passed=bool(ok), final_x=round(float(x[-1]), 3), dz=round(float(z[-1] - z[0]), 3),
                             max_tilt_deg=round(float(tilt.max()), 1))
        out[kind] = res
    rest = log['speed'][int(settle / PHYSICS_DT) - 1]
    out['rest_base_height_m'] = round(float(rest[3]), 4)
    out['rest_tilt_deg'] = round(float(rest[7]), 3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--urdf-dir', type=Path, required=True, help='the scout_front_camera package')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--usd-dir', type=Path, help='keep the converted USD here (the Isaac evaluation robot)')
    parser.add_argument('--torque-nm', type=float, default=MOTOR_W * .080 / TOP_SPEED_MPS,
                        help='wheel torque limit (default: rated power at top speed)')
    parser.add_argument('--friction', type=float, default=1.0, help='tyre-ground Coulomb friction (URDF Gazebo mu: 1.0)')
    args = parser.parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    from kappanav.common.physical import SCOUT_MINI
    work = Path(tempfile.mkdtemp())
    usd = convert(scout_urdf(args.urdf_dir, work, SCOUT_MINI.wheel_radius_m), args.usd_dir or work)
    sim, robots = build(usd, SCOUT_MINI.wheel_radius_m, args.torque_nm, args.friction)
    log, settle = run(sim, robots, SCOUT_MINI.wheel_radius_m, SCOUT_MINI.wheel_track_m)
    result = dict(summarize(log, settle), usd=str(usd), mass_kg=MASS_KG, friction=args.friction,
                  torque_limit_nm=args.torque_nm, wheel_damping=WHEEL_DAMPING, joints=robots['speed'].joint_names)
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    np.savez_compressed(args.output.with_suffix('.npz'), **log)
    print(json.dumps(result, indent=1), flush=True)
    if args.usd_dir is None:
        shutil.rmtree(work, ignore_errors=True)
    app.close()


if __name__ == '__main__':
    main()
