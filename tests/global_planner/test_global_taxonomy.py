import numpy as np
import pytest

from kappanav.global_planner.taxonomy import CLASSES, SOURCE_CLASSES, bits, label_table, route_cost, traversability


@pytest.mark.parametrize('source',sorted(SOURCE_CLASSES))
def test_every_mapping_uses_known_classes(source):
    for name,target in SOURCE_CLASSES[source].items():
        assert set(target)<=set(CLASSES),(source,name)


def test_label_table_requires_every_native_class_and_maps_sets():
    lut=label_table('potsdam',['impervious surfaces','building','low vegetation','tree','car','clutter/background'])
    assert lut[2]==bits(('dirt','grass','shrub')) and lut[0]==bits(('road',))
    with pytest.raises(KeyError):label_table('potsdam',['unknown thing'])


def test_an_unmapped_source_says_so_and_lists_its_native_classes():
    with pytest.raises(KeyError,match='no taxonomy mapping.*tile_a'):
        label_table('a_source_that_does_not_exist',['tile_a'])


def test_an_oblique_release_uses_its_nadir_sources_mapping():
    aeroscapes=['Background','Person','Bike','Car','Drone','Boat','Animal','Obstacle','Construction','Vegetation','Road','Sky']
    mavsd=['background','solidago canadensis','person','road','pine','bush','reed','stone','cypress','water','building',
           'vegetable garden','electric bicycle']
    assert (label_table('aeroscapes_oblique',aeroscapes)==label_table('aeroscapes',aeroscapes)).all()
    assert (label_table('mavsd_oblique',mavsd)==label_table('mavsd',mavsd)).all()


def test_every_store_source_that_labels_pixels_has_a_mapping():
    """The sources the data pipeline writes must all be mapped, or training stops at the first KeyError."""
    from kappanav.global_planner.prepare.__main__ import SOURCES
    missing=sorted(set(SOURCES)-set(SOURCE_CLASSES)-{'global_south'})
    assert not missing,missing


def test_uavid_supervises_its_vegetation_and_not_its_clutter():
    lut=label_table('uavid',['clutter','building','road','static car','tree','vegetation','human','moving car'])
    assert lut[5]==bits(('grass','shrub','field')) and lut[0]!=0 and lut[6]==0


def test_scores_report_dangerous_errors_and_calibration():
    from kappanav.global_planner.metrics import Scores
    from kappanav.global_planner.taxonomy import drivable_probability
    posterior=np.zeros((len(CLASSES),1,4),np.float32);posterior[CLASSES.index('road')]=1
    labels=np.array([[bits(('road',)),bits(('road',)),bits(('water',)),bits(('obstacle',))]])
    scores=Scores();scores.add(posterior,drivable_probability(posterior),traversability(posterior),labels);r=scores.report()
    assert r['false_passable']==1 and r['missed_passable']==0 and abs(r['calibration_error']-.5)<1e-6


def test_value_and_cost_follow_the_priorities_and_blend_smoothly():
    eye=np.eye(len(CLASSES),dtype=np.float32)[:,None]
    value=traversability(eye)[0]
    assert (np.diff(value)<=0).all() and value[0]==255 and value[CLASSES.index('obstacle')]==0
    assert (np.diff(route_cost(value))>=0).all() and route_cost(value)[0]==1
    half=traversability((eye[:,:,:1]+eye[:,:,2:3])/2)
    assert value[2]<half[0,0]<value[0]


def test_the_stored_sources_map_every_native_class():
    from kappanav.global_planner.dataset import EXCLUDED, STORE, Source
    if not STORE.is_dir():pytest.skip('no data store here')
    for folder in STORE.iterdir():
        if (folder/'manifest.jsonl').is_file() and folder.name not in EXCLUDED:Source(folder.name)
