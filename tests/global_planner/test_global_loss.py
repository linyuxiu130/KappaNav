import torch
from torch.nn import functional as F

from kappanav.global_planner.loss import SurfaceCriterion
from kappanav.global_planner.taxonomy import CLASSES, bits

K=len(CLASSES)


def test_single_class_labels_are_plain_cross_entropy():
    g=torch.Generator().manual_seed(3)
    logits=torch.randn(2,K,8,8,generator=g)
    target=torch.randint(K,(2,8,8),generator=g)
    loss=SurfaceCriterion()(logits,(1<<target).int())
    torch.testing.assert_close(loss,F.cross_entropy(logits,target))


def test_a_set_label_scores_the_probability_of_the_whole_set_and_ignore_gets_no_gradient():
    logits=torch.randn(1,K,1,2,requires_grad=True)
    labels=torch.tensor([[[bits(('shrub','tree')),0]]])
    loss=SurfaceCriterion()(logits,labels)
    p=logits.softmax(1)[0,:,0,0]
    torch.testing.assert_close(loss,-torch.log(p[CLASSES.index('shrub')]+p[CLASSES.index('tree')]))
    loss.backward()
    assert (logits.grad[...,1]==0).all()
    members=[CLASSES.index('shrub'),CLASSES.index('tree')];others=[k for k in range(K) if k not in members]
    inside,outside=logits.grad[0,members,0,0],logits.grad[0,others,0,0]
    assert (outside>0).all() and (inside<0).all()
    torch.testing.assert_close(inside[0],inside[1])                                # no member is favoured
    torch.testing.assert_close(inside.sum(),-outside.sum())                         # and the logits do not drift
