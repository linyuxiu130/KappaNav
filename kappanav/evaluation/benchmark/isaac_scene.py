"""The Isaac evaluation scene: an official scene USD, the Dingo with its depth camera and base contact sensor, PhysX at
100 Hz with a control step every 0.1 s, the official reset, time-out and latched arrival. Stepping follows Isaac Lab's
manager-based environment step for step. Import after Isaac Sim has started (isaaclab.app.AppLauncher)."""
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.sensors import CameraCfg,ContactSensorCfg
from isaaclab.sim.utils import attach_stage_to_usd_context,use_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from pathlib import Path
from pxr import Gf,PhysxSchema,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
from kappanav.common.physical import DINGO, ROBOT
from kappanav.evaluation.benchmark.protocol import GOAL_BUDGET_S
from kappanav.evaluation.simulation.realtime import GOAL_TOLERANCE_M, ARRIVAL_SPEED_MPS, ARRIVAL_HOLD_S

if ROBOT is not DINGO:
    # The wheel joints, camera prim and contact sensor below are the benchmark dingo.usd's; the Scout Mini's Isaac
    # model is driven by simulation.calibrate_scout_mini.
    raise NotImplementedError(f'the Isaac evaluation scene is set up for the Dingo, not {ROBOT.name}')

PHYSICS_DT,CONTROL_STEPS=.01,10          # one control step and one render per 0.1 s
SCENE_SCALE=.01                          # scene USDs are authored in centimetres
BLANKET_SCALE=.01                        # the official loader shrinks every blanket to 1 %
START_HEIGHT_M=.1                        # the robot is dropped from this height above the start
CAMERA_WIDTH,CAMERA_HEIGHT,CAMERA_FOCAL_PX,CAMERA_FOCAL_LENGTH=640,360,326.398559570312,1.93
WHEEL_JOINTS=['left_wheel_joint','right_wheel_joint']
COLLISION_CONTACT_OFFSET=.03             # contact offset of the scene's merged collision meshes
_MESH_COLLISION_SCHEMAS=(('PhysicsCollisionAPI',UsdPhysics.CollisionAPI),('PhysicsMeshCollisionAPI',UsdPhysics.MeshCollisionAPI),
    ('PhysxConvexHullCollisionAPI',PhysxSchema.PhysxConvexHullCollisionAPI),('PhysxConvexDecompositionCollisionAPI',PhysxSchema.PhysxConvexDecompositionCollisionAPI),
    ('PhysxSDFMeshCollisionAPI',PhysxSchema.PhysxSDFMeshCollisionAPI),('PhysxTriangleMeshCollisionAPI',PhysxSchema.PhysxTriangleMeshCollisionAPI),
    ('PhysxMeshMergeCollisionAPI',PhysxSchema.PhysxMeshMergeCollisionAPI))


@configclass
class _SceneCfg(InteractiveSceneCfg):
    terrain:TerrainImporterCfg=None
    robot:ArticulationCfg=None
    contact_sensor:ContactSensorCfg=None
    camera_sensor:CameraCfg=None


def _scene_cfg(scene_usd,robot_usd):
    robot=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
        spawn=sim_utils.UsdFileCfg(usd_path=str(robot_usd),activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,retain_accelerations=False,linear_damping=0.,
                angular_damping=0.,max_linear_velocity=1000.,max_angular_velocity=1000.,max_depenetration_velocity=1.)),
        actuators={'base':ImplicitActuatorCfg(joint_names_expr=WHEEL_JOINTS,velocity_limit=100.,effort_limit=20.,stiffness=0.,damping=1.)})
    aperture=lambda pixels:pixels*CAMERA_FOCAL_LENGTH/CAMERA_FOCAL_PX
    camera=CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/base_link/front_cam',update_period=.05,height=CAMERA_HEIGHT,width=CAMERA_WIDTH,
        data_types=['distance_to_image_plane'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAMERA_FOCAL_LENGTH,focus_distance=.205,horizontal_aperture=aperture(CAMERA_WIDTH),
            vertical_aperture=aperture(CAMERA_HEIGHT),clipping_range=(.01,100.)),
        offset=CameraCfg.OffsetCfg(pos=(.28618,0.,.62532),rot=(-.54168,-.45452,.45452,.54168),convention='usd'))
    return _SceneCfg(num_envs=1,env_spacing=0.,
        terrain=TerrainImporterCfg(prim_path='/World/Scene',terrain_type='usd',usd_path=str(scene_usd)),robot=robot,
        contact_sensor=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/base_link',history_length=10,track_air_time=True,update_period=.02),
        camera_sensor=camera)


class IsaacScene:
    step_dt=PHYSICS_DT*CONTROL_STEPS
    hold_steps=int(round(ARRIVAL_HOLD_S/(PHYSICS_DT*CONTROL_STEPS)))

    def __init__(self,scene_usd,robot_usd,device):
        self.sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT,render_interval=CONTROL_STEPS,device=device))
        stage=self.sim.get_initial_stage()
        with use_stage(stage):
            self.scene=InteractiveScene(_scene_cfg(scene_usd,robot_usd));attach_stage_to_usd_context()
            # The scene USD carries a physics scene of its own (CPU dynamics), kept as in the reference environment;
            # every physics scene steps at the simulation rate.
            share_materials(stage)
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):PhysxSchema.PhysxSceneAPI.Apply(prim).CreateTimeStepsPerSecondAttr().Set(round(1/PHYSICS_DT))
            self.sim.reset()
        self.scene.update(dt=PHYSICS_DT)
        terrain=stage.GetPrimAtPath('/World/Scene/terrain');_scale(terrain,SCENE_SCALE)
        for prim in Usd.PrimRange(terrain):
            if 'blanket' in prim.GetName().lower():
                for child in prim.GetAllChildren():_scale(child,BLANKET_SCALE)
        _merged_collision(terrain)
        self.stage=stage;self.physics_steps=0
        self.robot=self.scene['robot'];self.camera=self.scene.sensors['camera_sensor'];self.contact=self.scene.sensors['contact_sensor']
        self.wheels=self.robot.find_joints(WHEEL_JOINTS,preserve_order=True)[0]

    def reset(self,start,goal=None,budget_s=GOAL_BUDGET_S):
        """Drop the robot at start (x, y, yaw) and clear the episode clock and arrival latch; without a goal the
        episode runs until the budget."""
        x,y,yaw=start;device=self.robot.device
        self.scene.reset()
        pose=torch.tensor([[x,y,START_HEIGHT_M,np.cos(yaw/2),0.,0.,np.sin(yaw/2)]],dtype=torch.float32,device=device)
        self.robot.write_root_pose_to_sim(pose);self.robot.write_root_velocity_to_sim(self.robot.data.default_root_state[:,7:13])
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos.clone(),self.robot.data.default_joint_vel.clone())
        UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/envs/env_0/Robot')).MakeInvisible()
        self.scene.write_data_to_sim();self.sim.forward()
        self.goal=None if goal is None else torch.tensor(goal,dtype=torch.float32,device=device)
        self.max_steps=int(np.ceil(budget_s/self.step_dt-1e-9));self.steps=0;self.latched=False;self.held=0

    def close(self):
        """Release the scene and the simulation context in Isaac Lab's teardown order."""
        del self.robot,self.camera,self.contact,self.scene
        self.sim.clear_all_callbacks();self.sim.clear_instance()

    def step(self,command):
        """One control step of the differential drive [v, w]; returns running, goal_reached or budget_exhausted."""
        v,w=float(command[0]),float(command[1])
        track,radius=ROBOT.wheel_track_m,ROBOT.wheel_radius_m
        target=torch.tensor([[(2*v-w*track)/(2*radius),(2*v+w*track)/(2*radius)]],device=self.robot.device)
        for _ in range(CONTROL_STEPS):
            self.physics_steps+=1
            self.robot.set_joint_velocity_target(target,joint_ids=self.wheels)
            self.scene.write_data_to_sim();self.sim.step(render=False)
            if self.physics_steps%CONTROL_STEPS==0:self.sim.render()
            self.scene.update(dt=PHYSICS_DT)
        self.steps+=1
        # Terminations are evaluated before this step's arrival latch; a time-out takes precedence.
        if self.steps>=self.max_steps:return 'budget_exhausted'
        if self.goal is None:return 'running'
        if self.held>=self.hold_steps:return 'goal_reached'
        near=torch.linalg.norm(self.robot.data.root_pos_w[0,:2]-self.goal)<GOAL_TOLERANCE_M
        slow=self.robot.data.root_lin_vel_w[0].abs().sum()<ARRIVAL_SPEED_MPS
        if bool(near&slow):self.latched=True
        if self.latched:self.held+=1
        return 'running'


def _asset_key(value):
    if isinstance(value,Sdf.AssetPath):
        resolved=value.resolvedPath
        return str(Path(resolved).resolve(strict=True)) if Path(resolved).is_absolute() else resolved or value.path
    return str(value)


def share_materials(stage):
    """Bind every set of identical single-shader materials to one of them, with MDL paths in canonical form, so that
    each distinct material is loaded and compiled once; no material parameter changes."""
    for prim in stage.Traverse():
        asset=prim.GetAttribute('info:mdl:sourceAsset')
        if asset and asset.Get() is not None and Path(asset.Get().resolvedPath).is_absolute():
            asset.Set(Sdf.AssetPath(str(Path(asset.Get().resolvedPath).resolve(strict=True))))
    canonical,duplicates={},{}
    for prim in stage.Traverse():
        children=prim.GetChildren() if prim.IsA(UsdShade.Material) else []
        if len(children)!=1 or not children[0].IsA(UsdShade.Shader):continue
        shader=children[0]
        if any(a.GetConnections() for a in shader.GetAttributes()):continue
        key=(tuple((a.GetName(),str(a.GetTypeName()),_asset_key(a.Get())) for a in shader.GetAttributes()),
            tuple((a.GetName(),_asset_key(a.Get()),tuple(str(c).replace(str(shader.GetPath()),'SHADER') for c in a.GetConnections())) for a in prim.GetAttributes()))
        if key in canonical:duplicates[prim.GetPath()]=canonical[key]
        else:canonical[key]=prim.GetPath()
    for prim in stage.Traverse():
        for relationship in prim.GetRelationships():
            if relationship.GetName().startswith('material:binding'):
                targets=relationship.GetTargets();shared=[duplicates.get(t,t) for t in targets]
                if shared!=targets:relationship.SetTargets(shared)
    for path in duplicates:stage.GetPrimAtPath(path).SetActive(False)


def _merged_collision(prim):
    """Every prim that names a collision-mesh collection collides as one merged triangle mesh, with the reference
    scene loader's contact offset; the collection members are the collision geometry."""
    for child in prim.GetAllChildren():
        if child.GetAttribute('collection:collisionmeshes').IsValid():
            applied=child.GetAppliedSchemas()
            for name,schema in _MESH_COLLISION_SCHEMAS:
                if name in applied:child.RemoveAPI(schema)
            UsdPhysics.CollisionAPI.Apply(child);UsdPhysics.MeshCollisionAPI.Apply(child)
            PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(child);PhysxSchema.PhysxMeshMergeCollisionAPI.Apply(child)
            PhysxSchema.PhysxCollisionAPI.Apply(child).CreateContactOffsetAttr().Set(COLLISION_CONTACT_OFFSET)
        _merged_collision(child)


def _scale(prim,factor):
    scale=prim.GetAttribute('xformOp:scale')
    if not scale:scale=prim.CreateAttribute('xformOp:scale',Sdf.ValueTypeNames.Double3,False)
    scale.Set(Gf.Vec3d(factor,factor,factor))
    order=prim.GetAttribute('xformOpOrder')
    if not order.HasValue():prim.CreateAttribute('xformOpOrder',Sdf.ValueTypeNames.TokenArray,False).Set(['xformOp:translate','xformOp:orient','xformOp:scale'])
    elif 'xformOp:scale' not in list(order.Get()):order.Set(list(order.Get())+['xformOp:scale'])
