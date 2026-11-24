import torch
from test_critic import oracle,source_batch
from kappanav.local_planner.data.value_targets import RouteUtilityTeacher


def test_no_goal_utility_uses_causal_coverage_not_hidden_exit():
    batch=source_batch(1,'cpu');batch['goal_present'][:]=False
    t=torch.linspace(0,2,64);paths=torch.stack((torch.stack((t*0,t),-1),torch.stack((t*0,-t),-1)))[None]
    teacher=RouteUtilityTeacher(oracle(),3.6)
    batch['observed_field'][:,3,32:]=0
    scores=teacher(paths,batch).score
    assert scores[0,0]>scores[0,1]
    batch['observed_field'][:,3]=1-batch['observed_field'][:,3]
    assert teacher(paths,batch).score[0,0]<teacher(paths,batch).score[0,1]




def test_outside_coverage_lattice_does_not_create_a_novelty_bonus():
    batch=source_batch(1,'cpu');batch['goal_present'][:]=False
    t=torch.linspace(0,6,64);paths=torch.stack((t*0,t),-1)[None,None]
    labels=RouteUtilityTeacher(oracle(),3.6)(paths,batch)
    # All available coverage is known. Padding outside the local label field is not unseen evidence.
    from kappanav.local_planner.data.value_targets import in_place_turn_m
    torch.testing.assert_close(labels.score,(labels.progress_m-.5*(torch.tensor([[6.]])+in_place_turn_m(paths)))/3.6)
    assert labels.progress_m.item()<6.1
