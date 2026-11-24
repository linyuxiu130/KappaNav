"""A resume must reproduce the recipe but accept the provenance of a warm start and a cleaned store."""
import pytest

pytest.importorskip('transformers')

from kappanav.global_planner.train import same_recipe

CONFIG = {'sources': ['flair', 'messi'], 'source_weights': {'flair': .6, 'messi': .4}, 'steps': 40000,
          'batch_size_per_rank': 8, 'world_size': 4, 'lr': 6e-4, 'backbone_lr': 1e-5, 'seed': 42,
          'eval_every': 4000, 'eval_images': 300, 'torch': '2.7.1', 'init': '/w/earlier/best.pt',
          'source_sha256': {'a.py': 'before'}, 'train_rows': {'flair': 61712, 'messi': 684}}


def test_a_warm_started_run_resumes_into_the_config_it_would_build_now():
    """`--init` writes `init` into the config the run saves; the resumed run has no --init to rebuild it from."""
    resumed = {k: v for k, v in CONFIG.items() if k not in ('init', 'source_sha256', 'train_rows')}
    assert same_recipe(CONFIG, resumed)


def test_a_cleaned_store_does_not_block_a_resume():
    resumed = {**CONFIG, 'train_rows': {'flair': 61200, 'messi': 684}}
    assert same_recipe(CONFIG, resumed)


@pytest.mark.parametrize('key,value', [('sources', ['messi']), ('world_size', 8), ('lr', 1e-4), ('steps', 100),
                                       ('seed', 0), ('eval_every', 1000)])
def test_a_changed_recipe_blocks_it(key, value):
    assert not same_recipe(CONFIG, {**CONFIG, key: value})
