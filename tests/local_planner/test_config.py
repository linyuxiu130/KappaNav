import pytest

from kappanav.local_planner.model.config import CurveNavConfig
from kappanav.local_planner.model.config_io import config_from_mapping


def test_base_mapping_has_one_prepared_dataset_and_fixed_future_contract() -> None:
    config = config_from_mapping(
        {
            "data": {
                "root": "data/policy_dataset",
                "observation_frames": 4,
                "future_steps": 24,
            },
            "model": {
                "trajectory": {
                    "num_control_points": 8,
                },
                "trajectory_decoder": {"transformer_layers": 4},
            },
        }
    )
    assert config.data.root == "data/policy_dataset"
    assert config.data.observation_frames == 4
    assert config.data.future_steps == 24
    assert config.trajectory.num_control_points == 8
    assert config.trajectory_decoder.transformer_layers == 4


def test_rejects_removed_source_specific_data_config() -> None:
    with pytest.raises(TypeError, match="training_sources"):
        config_from_mapping({"data": {"training_sources": []}})
    with pytest.raises(TypeError, match="frame_skip"):
        config_from_mapping({"data": {"frame_skip": 2}})


@pytest.mark.parametrize('field,value',[
    ('generator_candidates',2),('cfm_times_per_step',1),('perception_gradient_fraction',.0625),
    ('trajectory_supervision_fraction',.0625),('compile_policy',True),('deterministic_algorithms',False)])
def test_formal_config_rejects_experimental_training_switches(field,value):
    with pytest.raises(TypeError,match=field):
        config_from_mapping({'training':{field:value}})


def test_rejects_removed_architecture_switches() -> None:
    with pytest.raises(TypeError, match="scale_xy"):
        config_from_mapping({"model": {"trajectory": {"scale_xy": [3.0, 3.0]}}})
    with pytest.raises(TypeError, match="normalization_scale_m"):
        config_from_mapping({"model": {"trajectory": {"normalization_scale_m": 4.0}}})
    with pytest.raises(TypeError, match="residual_increment_mean_xy_m"):
        config_from_mapping(
            {"model": {"trajectory": {"residual_increment_mean_xy_m": [0.0] * 14}}}
        )
    with pytest.raises(TypeError, match="maximum_curvature_inv_m"):
        config_from_mapping(
            {"model": {"trajectory": {"maximum_curvature_inv_m": 4.0}}}
        )
    with pytest.raises(TypeError,match='fine_token_budget'):
        config_from_mapping({'model':{'condition_encoder':{'fine_token_budget':128}}})


def test_rejects_removed_diagnostic_training_options() -> None:
    with pytest.raises(TypeError, match="overfit_steps"):
        config_from_mapping({"training": {"overfit_steps": 1}})
    with pytest.raises(TypeError, match="checkpoint_path"):
        config_from_mapping({"training": {"checkpoint_path": "unused.pt"}})


@pytest.mark.parametrize('key',['gradient_accumulation_steps','per_device_batch_size','samples_per_epoch',
    'num_workers','prefetch_factor','log_every_steps','output_dir','grad_clip_norm','micro_batch_size','global_batch_size'])
def test_rejects_training_knobs_that_the_official_trainer_does_not_execute(key):
    with pytest.raises(TypeError,match=key):config_from_mapping({'training':{key:2}})


def test_canonical_config_serialization_preserves_effective_settings():
    from kappanav.local_planner.model.config_io import config_to_mapping
    cfg=config_from_mapping({'model':{'depth_encoder':{'geometry_context_m':8.}},'training':{'epochs':4,'warmup_epochs':1}})
    assert config_from_mapping(config_to_mapping(cfg))==cfg


def test_defaults_have_one_training_budget_and_no_old_dataset_statistics():
    cfg=CurveNavConfig()
    assert (cfg.training.epochs,cfg.training.warmup_epochs,cfg.training.learning_rate)==(4,1,1e-3)
    assert cfg.trajectory.control_increment_mean_xy_m==(0.,)*14
    assert cfg.trajectory.control_increment_std_xy_m==(1.,)*14


def test_training_entry_rejects_a_missing_bundle_before_creating_output(monkeypatch,tmp_path):
    import sys
    from kappanav.local_planner.training.bspline_replay import main
    monkeypatch.setattr(sys,'argv',['train','--bundle','missing.pt','--output',str(tmp_path/'run')])
    with pytest.raises(SystemExit) as error:main()
    assert error.value.code==2 and not (tmp_path/'run').exists()


def test_rejects_invalid_trajectory_contract() -> None:
    with pytest.raises(TypeError, match="target_spline_degree"):
        config_from_mapping({"model": {"trajectory": {"target_spline_degree": 2}}})

    with pytest.raises(TypeError, match="flow_steps"):
        config_from_mapping(
            {"model": {"trajectory_decoder": {"flow_steps": 7}}}
        )
