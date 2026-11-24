"""The expert route at a state and its continuous geometric qualification as a B-spline teacher."""
import numpy as np
import torch
from kappanav.local_planner.data import bspline_replay as replay
from kappanav.local_planner.data.goal_distance import GoalDistanceQuery
import math
from kappanav.common.geometry import Grid, LABEL_CLEARANCE_M, SAFETY_STEP_M, PlanningError, _plan, _spatial_route
from kappanav.local_planner.model.trajectory.control_points import project_reference_tangent
from kappanav.local_planner.control.mpc import TURN_ENTER_RAD, TURN_LOOKAHEAD_M

# Arc length of the goal route that the receding-horizon expert smooths; well past the 3.6 m planning horizon.
REPLAN_REACH_M=5.


def _prefix(polyline,length):
    segments=np.linalg.norm(np.diff(polyline,axis=0),axis=1);arc=np.r_[0.,np.cumsum(segments)]
    if arc[-1]<=length:return polyline
    k=int(np.searchsorted(arc,length));fraction=(length-arc[k-1])/segments[k-1]
    return np.vstack((polyline[:k],polyline[k-1]+fraction*(polyline[k]-polyline[k-1])))


def _reach_route(grid,view,goal):
    """The clearance-aware A* route to the goal cut at REPLAN_REACH_M of arc (PlanningError where none).
    Simplification stops past that reach, which leaves this prefix unchanged."""
    simplified,_=_spatial_route(grid,np.asarray(view[:2],dtype=np.float64),np.asarray(goal,dtype=np.float64),REPLAN_REACH_M)
    return _prefix(simplified,REPLAN_REACH_M)


def start_heading(view,reach):
    """The heading the expert curve leaves the state along: the body heading, unless the route's direction at
    TURN_LOOKAHEAD_M is TURN_ENTER_RAD or more off it. There the controller turns in place first (simulation.mpc,
    same lookahead and threshold) and the curve starts along the route; a circular body turning on the spot
    stays where its start is. A forward curve from the body heading would be a U-turn loop the controller does
    not drive as planned, longer than the route and the most expensive curve to plan."""
    reach=np.asarray(reach,dtype=np.float64);heading=float(view[2])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reach,axis=0),axis=1))]
    if arc[-1]<=1e-9:return heading
    ahead=np.array([np.interp(min(TURN_LOOKAHEAD_M,arc[-1]),arc,reach[:,axis]) for axis in range(2)])-reach[0]
    direction=float(np.arctan2(ahead[1],ahead[0]))
    return direction if abs(math.remainder(direction-heading,2*math.pi))>=TURN_ENTER_RAD else heading


def _forward_plan(grid,view,reach):
    try:return _plan(grid,reach,start_heading(view,reach))
    except PlanningError:return None


def _forward(grid,view,reach):
    plan=_forward_plan(grid,view,reach)
    return None if plan is None else plan.path_xy


def forward_route(grid,view,goal):
    """The clearance-aware route to the goal, cut at REPLAN_REACH_M of arc, then smoothed and certified from
    start_heading by the capture planner. Its first 3.6 m matches the full-route plan (median max deviation
    2 mm) and, as a B-spline teacher, the executed MPC future (8 mm). None where no forward curve certifies."""
    try:reach=_reach_route(grid,view,goal)
    except PlanningError:return None
    return _forward(grid,view,reach)


def _expert_from_reach(grid,view,reach):
    """expert_route given the goal's _reach_route: the forward curve where one certifies, else that polyline."""
    route=_forward(grid,view,reach)
    return reach if route is None else route


def expert_route(grid,view,goal):
    """The expert's world route from a state, the same for every capture whoever drove it.

    The forward route where one certifies from start_heading (along the route where it turns TURN_ENTER_RAD or
    more off the body heading: the controller turns in place first). Otherwise (facing, next to or touching an
    obstacle) the escape route: the same clearance-aware A* polyline, unsmoothed and without a heading
    constraint, so the controller first turns toward it. Its fitted curve is certified by the escape rule in
    qualified_route_teacher. A local exit is also available when an endpoint cannot be snapped.
    """
    # The A* route is computed once: forward_route's plan and the unsmoothed fallback share it.
    try:reach=_reach_route(grid,view,goal)
    except PlanningError:return next(escape_routes(grid,np.asarray(view[:2]),np.asarray(goal)),None)
    return _expert_from_reach(grid,view,reach)


def escape_routes(grid,start,goal):
    """Local straight exits, ordered by exit length plus remaining goal distance.

    A recovery label ends at a safe waypoint; subsequent observations replan toward the mission.
    An invalid/disconnected goal does not prevent leaving contact, but stays unavailable to Q.
    """
    if grid.boundary_distance(start[None])[0][0]>=LABEL_CLEARANCE_M:return
    reach=1.
    lower=np.maximum(grid.world_to_grid(start-reach),0)
    upper=np.minimum(grid.world_to_grid(start+reach)+1,grid.free.shape)
    cells=np.argwhere(grid.free[lower[0]:upper[0],lower[1]:upper[1]])+lower
    if not len(cells):return
    points=grid.grid_to_world(cells);length=np.linalg.norm(points-start,axis=1)
    keep=(length<=reach)&(grid.boundary_distance(points)[0]>=LABEL_CLEARANCE_M)
    points,cells,length=points[keep],cells[keep],length[keep]
    try:target=grid.snap(goal)
    except PlanningError:remaining=np.linalg.norm(points-goal,axis=1)
    else:
        from kappanav.common.grid_dijkstra import grid_distances
        distances=grid_distances(grid.free,grid.cell_size_m,int(target[0]),int(target[1]),cells)
        remaining=distances[cells[:,0],cells[:,1]]
        if not np.isfinite(remaining).any():remaining=np.linalg.norm(points-goal,axis=1)
    for index in np.argsort(length+remaining,kind='stable'):
        if not np.isfinite(remaining[index]):continue
        line=np.linspace(start,points[index],max(2,math.ceil(length[index]/(SAFETY_STEP_M/2))+1))
        if escape_clear(grid.boundary_distance(line)[0],LABEL_CLEARANCE_M):
            yield np.stack((start,points[index]))


def pack_routes(routes):
    """A capture's anchor routes (world polylines, None where expert_route found none) as flat arrays."""
    lengths=[0 if route is None else len(route) for route in routes]
    points=[np.asarray(route,dtype=np.float32) for route in routes if route is not None]
    return dict(expert_routes=np.concatenate(points) if points else np.zeros((0,2),np.float32),
                expert_route_offsets=np.r_[0,np.cumsum(lengths)].astype(np.int64),expert_route_valid=np.array([r is not None for r in routes]))


def unpack_routes(raw):
    """The anchor routes pack_routes stored in a capture."""
    points,offsets,valid=raw['expert_routes'],raw['expert_route_offsets'],raw['expert_route_valid']
    return [points[offsets[i]:offsets[i+1]] if ok else None for i,ok in enumerate(valid)]


ESCAPE_TOLERANCE_M=1e-3


def escape_clear(clearance,pad):
    """Clear by `pad` from the first sample that is, and never closer to an obstacle before it.

    For a curve starting clear this is the plain margin check; from a start next to or inside an obstacle it
    admits exactly the curves that only move away from it until they are clear (no sample gets closer).
    """
    clearance=np.asarray(clearance,dtype=np.float64);reached=np.flatnonzero(clearance>=pad)
    if not len(reached):return False
    first=reached[0]
    if first==0:return bool(np.all(clearance>=pad))
    prefix=clearance[:first+1]
    return bool(np.all(np.maximum.accumulate(prefix)-prefix<=ESCAPE_TOLERANCE_M) and np.all(clearance[first:]>=pad))


def _fit_route_teacher(actor,view,pose,route,tree,error,source_query):
    """Fit and certify an existing world route with the expert safety checks (escape_clear on both geometries)."""
    reference=replay.resample_route(route,horizon=actor.policy.planning_horizon_m)
    if reference is None:return None,dict(reason='zero_length_route')
    local=(reference-pose[:2,3])@pose[:2,:2];local[0]=0
    device=next(actor.policy.parameters()).device
    values,_=project_reference_tangent(actor.policy.curve_codec,torch.tensor(local,dtype=torch.float32,device=device)[None])
    dense,pad=replay.dense_curve(values);pad=float(pad[0]);dense=dense[0].cpu().numpy()
    world=dense@pose[:2,:2].T+pose[:2,3]
    observed_clearance=tree.query(dense,workers=1)[0]-error-actor.head.radius
    source_clearance=source_query.signed_distance(torch.from_numpy(world),0).cpu().numpy()
    observed=float(observed_clearance.min());source_minimum=float(source_clearance.min())
    checks=dict(observed_hit_clear=escape_clear(observed_clearance,pad),source_margin=escape_clear(source_clearance,pad))
    certificate=dict(kind='continuous_spline_geometry',qualified=all(checks.values()),observed_margin_m=observed,
        source_margin_m=source_minimum,sampling_pad_m=pad)
    if not certificate['qualified']:return None,dict(reason='geometric_curve_qualification',checks=checks,certificate=certificate)
    return dict(teacher_values=values[0].cpu(),teacher_checks=checks,teacher_certificate=certificate),None


@torch.no_grad()
def qualified_route_teacher(actor,view,pose,route,tree,error,source_query):
    """Keep the mission fit when certified; otherwise certify a local contact exit."""
    teacher,rejection=_fit_route_teacher(actor,view,pose,route,tree,error,source_query)
    if teacher is not None:return teacher,None
    source=source_query.grids[0]
    grid=Grid(source.signed_clearance_m>=0,np.maximum(source.signed_clearance_m,0),source.origin_xy,source.cell_size_m)
    for exit_route in escape_routes(grid,np.asarray(view[:2]),np.asarray(route[-1])):
        teacher,_=_fit_route_teacher(actor,view,pose,exit_route,tree,error,source_query)
        if teacher is not None:
            teacher['teacher_certificate']['recovery_waypoint_world_xy']=exit_route[-1].tolist()
            return teacher,None
    return None,rejection


def source_utility_availability(goal_distance: GoalDistanceQuery, view, point_goal):
    """Use exactly RouteUtilityTeacher's fixed source cells before declaring Q available."""
    source_query=goal_distance.source
    goal=torch.as_tensor(point_goal,dtype=torch.float32)[None]
    cells=source_query.point_cells(torch.stack((torch.zeros_like(goal),goal),1),
        torch.zeros(1,dtype=torch.long),torch.as_tensor(view[:2],dtype=torch.float64)[None],
        torch.tensor([view[2]],dtype=torch.float64))[0].cpu().numpy()
    grid=source_query.grids[0]
    inside=(cells>=0).all(1)&(cells<grid.signed_clearance_m.shape).all(1)
    traversable=bool(inside.all() and (grid.signed_clearance_m[cells[:,0],cells[:,1]]>=0).all())
    available=traversable and goal_distance.connected(0,cells[0],cells[1])
    return available,dict(defined=available,start_cell=cells[0].tolist(),goal_cell=cells[1].tolist(),
        reason=None if available else 'source_start_or_goal_disconnected',
        semantics='RouteUtilityTeacher point_cells, connected as goal_distance distances are finite; no snapping or source graph modification')


# No-goal teachers: one candidate target per heading sector, at the end of the expert's smoothing reach.
EXPLORATION_SECTORS=8
EXPLORATION_TEACHERS=2

@torch.no_grad()
def _body_samples(polyline,view,count=64):
    """Arc-uniform samples of a world polyline in the body frame at `view`."""
    segments=np.linalg.norm(np.diff(polyline,axis=0),axis=1);arc=np.r_[0.,np.cumsum(segments)]
    at=np.linspace(0.,arc[-1],count)
    xy=np.column_stack((np.interp(at,arc,polyline[:,0]),np.interp(at,arc,polyline[:,1])))-np.asarray(view[:2])
    c,s=np.cos(float(view[2])),np.sin(float(view[2]))
    return xy@np.array([[c,-s],[s,c]])


_UNSET=object()


def reachable_cells(grid,view):
    """The free cells reachable from the state's cell, as a mask (None off the grid or outside free space). Only
    reachability is needed (relabelled goals and exploration targets use Euclidean ranges), and grid_distances'
    diagonal rule makes it four-connected: one cached component labelling (Grid.components) replaces a full-scene
    Dijkstra per extra anchor (8% of a large scene's main-process time)."""
    here=grid.world_to_grid(np.asarray(view[:2],dtype=np.float64)[None])[0]
    if (here<0).any() or (here>=grid.free.shape).any() or not grid.free[tuple(here)]:return None
    return grid.components==grid.components[tuple(here)]


def exploration_targets(actor,grid,view,reach=_UNSET):
    """One candidate target per heading sector: the reachable free cell nearest to the middle of the ring between
    the planning horizon and REPLAN_REACH_M ([] where the state is off the grid or nothing is reachable there).
    `reach` is reachable_cells(grid,view) where the caller already has it."""
    if reach is _UNSET:reach=reachable_cells(grid,view)
    if reach is None:return []
    cells=np.argwhere(reach);world=grid.grid_to_world(cells)
    offset=world-np.asarray(view[:2]);distance=np.linalg.norm(offset,axis=1)
    horizon=actor.policy.planning_horizon_m
    ring=(distance>=horizon)&(distance<=REPLAN_REACH_M)
    if not ring.any():return []
    bearing=(np.arctan2(offset[:,1],offset[:,0])-float(view[2]))%(2*np.pi)
    sector=np.minimum((bearing/(2*np.pi/EXPLORATION_SECTORS)).astype(int),EXPLORATION_SECTORS-1)
    middle=.5*(horizon+REPLAN_REACH_M);targets=[]
    for s in range(EXPLORATION_SECTORS):
        members=np.flatnonzero(ring&(sector==s))
        if len(members):targets.append(world[members[np.argmin(np.abs(distance[members]-middle))]])
    return targets


def exploration_ranking(actor,grid,view,utility,field,targets=_UNSET):
    """The sectors' A* routes cut at REPLAN_REACH_M, ranked best first by the no-goal utility of their first
    planning horizon, and the utility batch; None without a routable sector. `targets`: exploration_targets."""
    from kappanav.local_planner.data.replay_targets import critic_source_inputs
    if targets is _UNSET:targets=exploration_targets(actor,grid,view)
    if not targets:return None
    horizon=actor.policy.planning_horizon_m
    item=dict(source_grid_index=0,view=np.asarray(view,dtype=np.float64),goal=torch.zeros(2),local_target_field=field)
    batch=dict(critic_source_inputs(item),goal_present=torch.zeros(1,dtype=torch.bool))
    reaches=[];proxies=[]
    for target in targets:
        try:simplified=_spatial_route(grid,np.asarray(view[:2],dtype=np.float64),target,REPLAN_REACH_M)[0]
        except PlanningError:continue
        reaches.append(_prefix(simplified,REPLAN_REACH_M));proxies.append(_body_samples(_prefix(simplified,horizon),view))
    if not reaches:return None
    proxy=utility(torch.from_numpy(np.stack(proxies)).float()[None],batch).score[0]
    return reaches,torch.argsort(proxy,descending=True).tolist(),batch


def exploration_teachers(actor,grid,view,pose,tree,error,utility,field,*,keep=EXPLORATION_TEACHERS,ranking=_UNSET,routes=None):
    """Teachers for a state without a mission, chosen by the same utility Q learns for it.

    In every heading sector the reachable free cell nearest to the ring between the planning horizon and
    REPLAN_REACH_M is a candidate target; its expert_route is fitted and certified exactly like a goal teacher.
    The certified curves are ranked by RouteUtilityTeacher without a goal (geodesic displacement weighted by
    unobserved coverage, length, contact and clearance) and the best `keep` are returned, best first.
    """
    # Rank the sectors by the same utility on their A* polylines (cheap), then smooth and certify in that order
    # until `keep` teachers certify; the SLSQP fit is the labelling cost, so the other sectors are never fitted.
    # Each sector's A* route is kept for its expert_route below (same route, computed once).
    # `ranking` (exploration_ranking's result) and `routes` (index -> _expert_from_reach(...)) may be precomputed
    # by capture helpers; a route not among them is computed here.
    if ranking is _UNSET:ranking=exploration_ranking(actor,grid,view,utility,field)
    if ranking is None:return []
    reaches,order,batch=ranking
    teachers=[]
    for index in order:
        route=routes[index] if routes is not None and index in routes else _expert_from_reach(grid,view,reaches[index])
        teacher=None if route is None else qualified_route_teacher(actor,view,pose,route,tree,error,utility.source_query)[0]
        if teacher is not None:teachers.append(teacher)
        if len(teachers)==keep:break
    if not teachers:return []
    values=torch.stack([teacher['teacher_values'] for teacher in teachers])
    codec=actor.policy.curve_codec
    paths,_=codec.decode_values(values.to(codec.basis.device,codec.basis.dtype))
    score=utility(paths.float().cpu()[None],batch).score[0]
    order=torch.argsort(score,descending=True)[:keep].tolist()
    return [teachers[i] for i in order]


# Euclidean ranges of the extra goals given to every anchor. One goal per state makes the goal unidentifiable
# (p(route|obs,goal)=p(route|obs)); near, mid and far goals of the same observation make it the deciding input.
RELABEL_GOAL_BINS_M=((1.5,3.5),(4.,8.),(8.,25.))


def relabelled_goals(grid,view,rng,reach=_UNSET):
    """One reachable free goal per RELABEL_GOAL_BINS_M range, as world XY (bins without a reachable cell are skipped).
    `reach` is reachable_cells(grid,view) where the caller already has it."""
    if reach is _UNSET:reach=reachable_cells(grid,view)
    if reach is None:return []
    cells=np.argwhere(reach);world=grid.grid_to_world(cells)
    distance=np.linalg.norm(world-np.asarray(view[:2]),axis=1);goals=[]
    for low,high in RELABEL_GOAL_BINS_M:
        members=np.flatnonzero((distance>=low)&(distance<=high))
        if len(members):goals.append(world[members[rng.integers(len(members))]])
    return goals
