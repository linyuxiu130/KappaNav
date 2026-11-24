import torch
from kappanav.local_planner.training.bspline_objective import relative_utility_loss


def test_scene_offsets_and_zero_weight_padding_do_not_change_preferences():
    scores=torch.tensor([[2.,1.,0.,1000.]],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([[-2.,2.,0.,-1000.]],dtype=torch.float64)
    weight=torch.tensor([[.25,.5,.25,0.]],dtype=torch.float64)
    loss=relative_utility_loss(scores,target,weight)
    shifted=relative_utility_loss(scores+100.,target-37.,weight)
    torch.testing.assert_close(loss,shifted,rtol=0,atol=1e-12)
    gradient=torch.autograd.grad(loss,scores)[0]
    assert gradient[0,0]>0 and gradient[0,1]<0 and gradient[0,3]==0
    torch.testing.assert_close(gradient.sum(),torch.tensor(0.,dtype=torch.float64),rtol=0,atol=1e-12)


def test_correct_utility_gaps_have_zero_loss_at_any_absolute_score_level():
    target=torch.tensor([[-3.,1.,.5]],dtype=torch.float64)
    weight=torch.tensor([[.25,.375,.375]],dtype=torch.float64)
    scores=(target+123.).requires_grad_()
    loss=relative_utility_loss(scores,target,weight)
    assert loss<1e-24 and scores.argmax()==target.argmax()
    torch.testing.assert_close(torch.autograd.grad(loss,scores)[0],torch.zeros_like(scores),rtol=0,atol=1e-12)


def test_unavailable_Q_row_has_no_loss_or_gradient():
    scores=torch.tensor([[1.,2.,3.]],requires_grad=True)
    loss=relative_utility_loss(scores,torch.zeros_like(scores),torch.zeros_like(scores))
    assert loss==0
    torch.testing.assert_close(torch.autograd.grad(loss,scores)[0],torch.zeros_like(scores),rtol=0,atol=0)


def test_clearance_shortfall_charges_staying_in_contact_not_leaving_it():
    import torch
    from kappanav.local_planner.data.value_targets import clearance_shortfall
    from kappanav.common.physical import EXTRA_CLEARANCE_M as pad
    active=torch.ones(1,5,dtype=torch.bool)
    shortfall=lambda values:clearance_shortfall(torch.tensor([values],dtype=torch.float32),active).item()
    assert shortfall([0.,.03,.06,.1,.3])==0                          # escapes monotonically
    assert abs(shortfall([0.,0.,0.,0.,0.])-pad)<1e-7                 # slides along the wall
    assert abs(shortfall([0.,.05,.02,.2,.3])-.03)<1e-6               # loses clearance it had reached
    # From a clear start it is the plain margin penalty.
    clearance=torch.rand(200,30)*.4;clearance[:,0]=pad+torch.rand(200)*.3
    mask=torch.rand(200,30)>.1;mask[:,0]=True
    plain=torch.relu(pad-clearance.masked_fill(~mask,torch.inf).amin(-1))
    assert torch.equal(clearance_shortfall(clearance,mask),plain)
