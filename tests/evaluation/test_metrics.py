import numpy as np
import pytest
from kappanav.evaluation.benchmark.metrics import goal_metrics, exploration_metrics


def test_navigation_spl_counts_success_and_travel():
    assert goal_metrics(True,50.,75.)['spl']==pytest.approx(2/3)
    assert goal_metrics(False,50.,20.)['spl']==0.
    assert goal_metrics(True,50.,49.6)['spl']==1.


def test_exploration_accepts_backtracking_plateaus_and_compares_equal_area_time():
    times=[0.,10.,20.,30.];areas=[0.,10.,10.,20.]
    same=exploration_metrics(times,areas,times,areas,40.,False)
    assert same['coverage']==.5 and same['exploration_spl']==1.
    slower=exploration_metrics([0.,20.,30.,60.],areas,times,areas,40.,False)
    assert slower['exploration_spl']==.5
    partial=exploration_metrics(times,[0.,5.,5.,10.],times,areas,40.,False)
    assert partial['exploration_spl']==pytest.approx(.5*10/30)
    assert exploration_metrics(times,areas,times,areas,40.,True)['exploration_spl']==0.
    assert exploration_metrics(times,[0.,0.,0.,0.],times,areas,40.,False)['exploration_spl']==0.


def test_offline_candidates_distinguish_generation_selection_and_publication(tmp_path):
    from kappanav.evaluation.simulation.navigation import NavigationGrid
    from kappanav.evaluation.benchmark.trajectory_metrics import trajectory_metrics, aggregate_trajectory_metrics
    free=np.ones((10,10),bool);free[4,:]=False
    path=tmp_path/'grid.npz'
    np.savez(path,free=free,clearance_m=free.astype(np.float32),origin_xy=np.array([-3.,-3.]),cell_size_m=1.,coordinate_frame='canonical_world_xy')
    navigation=NavigationGrid(path)
    x=np.linspace(0,2,64,dtype=np.float32);zero=np.zeros_like(x)
    paths=np.stack((np.stack((x,zero),-1),np.stack((zero,x),-1)))*1.
    bank=np.tile(paths,(8,1,1))
    clear=np.zeros((2,16),bool);clear[0]=True  # the field shows every first-decision candidate clear
    raw=dict(candidate_paths=np.stack((bank,bank)),acquired_views=np.zeros((2,3)),selected_indices=np.array([0,1]),candidate_clear=clear)
    result=trajectory_metrics(navigation,raw,[dict(applied=True),dict(applied=False)],'cpu')
    assert result==dict(decisions=2,candidates=32,candidate_collisions=16,selected_collisions=1,no_safe_candidate_decisions=0,
        field_missed_candidate_collisions=8,field_missed_selected_collisions=1,no_field_clear_candidate_decisions=1,
        published_decisions=1,published_selected_collisions=1)
    total=aggregate_trajectory_metrics([result,result])
    assert total['candidate_collision_rate']==.5 and total['published_selected_collision_rate']==1.
    assert total['field_missed_candidate_rate']==.5 and total['no_field_clear_candidate_rate']==.5
    assert total['selection_error_with_safe_candidate_count']==2 and total['selection_error_with_safe_candidate_rate']==.5


def test_offline_candidates_accept_a_final_unpublished_decision(tmp_path):
    """A controller failure ends run_episode after the decision's bank is recorded but before its plan is appended,
    so the episode has one more decision than plans; that last decision counts as unpublished."""
    from kappanav.evaluation.simulation.navigation import NavigationGrid
    from kappanav.evaluation.benchmark.trajectory_metrics import trajectory_metrics
    free=np.ones((10,10),bool);free[4,:]=False
    path=tmp_path/'grid.npz'
    np.savez(path,free=free,clearance_m=free.astype(np.float32),origin_xy=np.array([-3.,-3.]),cell_size_m=1.,coordinate_frame='canonical_world_xy')
    navigation=NavigationGrid(path)
    x=np.linspace(0,2,64,dtype=np.float32);zero=np.zeros_like(x)
    bank=np.tile(np.stack((np.stack((x,zero),-1),np.stack((zero,x),-1))),(8,1,1))
    raw=dict(candidate_paths=np.stack((bank,bank)),acquired_views=np.zeros((2,3)),selected_indices=np.array([0,0]),candidate_clear=np.ones((2,16),bool))
    result=trajectory_metrics(navigation,raw,[dict(applied=True)],'cpu')
    assert result['decisions']==2 and result['selected_collisions']==2
    assert result['published_decisions']==1 and result['published_selected_collisions']==1


def test_colliding_expert_is_an_invalid_reference_without_losing_agent_coverage():
    result=exploration_metrics([0.,10.],[0.,10.],[0.,5.],[0.,5.],40.,False,expert_collision=True)
    assert result['coverage']==.25 and not result['expert_valid']
    assert result['exploration_spl'] is None and result['relative_area_time_efficiency'] is None


def test_aggregate_exploration_score_requires_every_expert_reference():
    from kappanav.evaluation.benchmark.metrics import summarize
    generation=dict(decisions=1,candidates=16,candidate_collisions=0,selected_collisions=0,no_safe_candidate_decisions=0,
        field_missed_candidate_collisions=0,field_missed_selected_collisions=0,no_field_clear_candidate_decisions=0,published_decisions=1,published_selected_collisions=0)
    def row(source,valid):
        return dict(source=source,goal=dict(success=True,spl=1.),exploration=dict(expert_valid=valid,exploration_spl=1. if valid else None,coverage=.5,new_area_m2=2.,explored_area_m2=3.,expert_area_m2=4.),
            goal_trajectory=generation,exploration_trajectory=generation,goal_collision=False,exploration_collision=False,
            goal_status='goal_reached',exploration_status='budget_exhausted',expert_status='budget_exhausted' if valid else 'collision',expert_collision=not valid)
    assert summarize([row('benchmark',True)]*2)['benchmark']['exploration_spl']==1.
    result=summarize([row('benchmark',False),row('benchmark',True)])
    assert result['benchmark']['exploration_spl'] is None and result['all']['exploration_spl'] is None
    assert result['all']['success_rate']==1. and result['all']['exploration_reference_valid_episodes']==1


def test_controller_failure_invalidates_reference_and_failed_agent_cannot_score_success():
    args=([0.,10.],[0.,10.],[0.,10.],[0.,10.],40.,False)
    assert exploration_metrics(*args,expert_failure=True)['exploration_spl'] is None
    assert not exploration_metrics(*args,expert_failure=True)['expert_valid']
    assert exploration_metrics(*args,failure=True)['exploration_spl']==0.


def test_navigation_only_rows_summarize_without_exploration():
    from kappanav.evaluation.benchmark.metrics import summarize
    generation=dict(decisions=1,candidates=16,candidate_collisions=0,selected_collisions=0,no_safe_candidate_decisions=0,
        field_missed_candidate_collisions=0,field_missed_selected_collisions=0,no_field_clear_candidate_decisions=0,published_decisions=1,published_selected_collisions=0)
    row=lambda success:dict(source='benchmark',goal=dict(success=success,spl=float(success)),goal_trajectory=generation,goal_collision=not success,goal_status='goal_reached' if success else 'collision')
    result=summarize([row(True),row(False)])['all']
    assert result['success_rate']==.5 and result['goal_collision_rate']==.5 and 'coverage' not in result
