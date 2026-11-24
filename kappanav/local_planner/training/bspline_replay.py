"""The one training run (docs/ARCHITECTURE.md section 3): `python -m kappanav.local_planner.training.bspline_replay`.

main() resolves the run's arguments (or an interrupted run's, with --resume) and starts one train_worker per GPU;
train_worker loads the shared replay (shared_replay), configures the model for training (configure_training_actor),
and runs training_loop: per step a task-balanced global batch (balanced_batch_indices), split into microbatches
whose five losses (bspline_objective.batched_losses) are backpropagated with gradient accumulation, the gradients
averaged across ranks, AdamW with a warmup-cosine schedule and an EMA of every parameter. Every epoch it validates
the EMA weights (convergence) and saves a checkpoint with the complete resume state.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import argparse,gc,json,os,time,random,resource,tempfile
import multiprocessing as mp
import numpy as np
import torch
import torch.distributed as dist
from kappanav.common import experiment_io
from kappanav.local_planner.model.config_io import config_to_mapping, config_from_mapping
from kappanav.common.experiment_io import sha
from kappanav.local_planner.inference.bundle import load_bundle, save_bundle, CANDIDATES, FORMAT as BUNDLE_FORMAT
from kappanav.local_planner.model.network.policy import INTEGRATION_STEPS
from kappanav.local_planner.data.bspline_dataset import SAMPLING, TRAIN_SAMPLING, sample_index, dataset_manifests
from kappanav.local_planner.training.bspline_objective import batched_losses, OBJECTIVE
from kappanav.local_planner.training.batches import prepare_training_batch


PROTOCOL='native_bspline_fourframe_direct_joint'
SOURCE_ROOT=Path(__file__).resolve().parents[2]
SOURCE_FILES=tuple(str(path.relative_to(SOURCE_ROOT)) for path in sorted(SOURCE_ROOT.rglob('*.py')))

def capture_rng(device):
    """This process's torch, CUDA, NumPy and Python RNG states (restored on resume)."""
    return dict(torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        numpy=np.random.get_state(), python=random.getstate())

def average_gradients(parameters, world):
    """Average the gradients over ranks in one flat all-reduce (missing gradients count as zero); returns their norm."""
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in parameters])
    if world>1:
        dist.all_reduce(flat)
        flat.div_(world)
    offset=0
    for parameter in parameters:
        parameter.grad=flat[offset:offset+parameter.numel()].reshape_as(parameter)
        offset+=parameter.numel()
    return flat.norm()


def step_statistics(terms,value,grad_norm,local_batch,global_batch,world):
    """Reduce losses on CUDA and check losses/gradients in one host transfer before AdamW."""
    totals=torch.stack([value.detach().double(),*[v.detach().double()*local_batch for v in terms.values()]])
    if world>1:dist.all_reduce(totals)
    stats=torch.cat((grad_norm.detach().double()[None],totals)).cpu()
    if not torch.isfinite(stats).all():raise FloatingPointError('Non-finite joint loss or averaged gradient; optimizer update rejected')
    return float(stats[0]),stats[2:]/global_batch



def rank_batch_sizes(world,batch_per_rank,global_batch=None):
    """Rows per rank of a global batch (default world*batch_per_rank), the remainder on the first ranks."""
    total=world*batch_per_rank if global_batch is None else global_batch
    count,remainder=divmod(total,world)
    return [count+(rank<remainder) for rank in range(world)]


def batch_indices(cases,outer,data_start,rank,world,batch_per_rank,*,global_batch=None,sampling_seed=None):
    """Rank `rank`'s slice of global batch `data_start+outer` in the trial-uniform sampling order (sample_index)."""
    sizes=rank_batch_sizes(world,batch_per_rank,global_batch)
    offset=sum(sizes[:rank])
    return [sample_index(cases,(data_start+outer)*sum(sizes)+offset+slot,seed=sampling_seed)
        for slot in range(sizes[rank])]


def balanced_batch_indices(cases, outer, data_start, rank,
                           world, batch_per_rank, source_keys,*,global_batch=None,sampling_seed=None):
    """Keep the uniform-task global batch, assigning shared frames to one rank."""
    quotas = rank_batch_sizes(world,batch_per_rank,global_batch)
    global_items = []
    for source_rank in range(world):
        for index in batch_indices(
                cases, outer, data_start, source_rank,
                world, batch_per_rank,global_batch=global_batch,sampling_seed=sampling_seed):
            global_items.append((len(global_items), index, source_keys[index]))

    rank_keys = [set() for _ in range(world)]
    assigned = [[] for _ in range(world)]
    for order, index, row_keys in sorted(
            global_items, key=lambda item: (-len(item[2]), item[0])):
        candidates = [r for r in range(world) if len(assigned[r]) < quotas[r]]
        def work(r):
            added=len(row_keys-rank_keys[r])
            return added,len(rank_keys[r])+added,r
        target = min(candidates, key=work)
        rank_keys[target].update(row_keys)
        assigned[target].append((order, index))
    return [index for _, index in sorted(assigned[rank])]


FULL_FORMAT='native_batched_joint_full_training'


def training_loop(actor,parameters,optimizer,scheduler,ema,load_batch,updates,*,local_batch,microbatch_size,world,
                  global_batch,cnn_batch_size,loader,on_step):
    """Optimization steps; `loader` builds the next microbatch's pinned host inputs while the current one
    backpropagates: microbatch i+1 of the same batch, or after the last one the next batch (records, plans and
    its first microbatch) while step k finishes and updates. With gradient accumulation, preparing a whole batch
    behind only the last backward left ~0.3 s per extra microbatch exposed.

    Batch preparation shares no mutable state with the forward: prepared microbatches own their host inputs, and
    the derived plan-input cache is locked. on_step(step,grad_norm,mean_losses,learning_rates,drain) logs and checkpoints;
    drain() waits for the loader to be idle (validation reads the same replay); a true return stops early.
    """
    pending=loader.submit(load_batch,0)
    drain=lambda:loader.submit(int).result()
    for outer in range(updates):
        batch=pending.result()
        optimizer.zero_grad(set_to_none=True);term_sums={}
        micros=batch.microbatches();ready=None
        for index,micro in enumerate(micros):
            if ready is not None:ready.result()
            terms=batched_losses(actor,micro,cnn_batch_size=cnn_batch_size)[0]
            value=sum(terms.values())*(world*len(micro)/global_batch)
            # Prepare during the backward only: the loader is Python-heavy and shares the GIL, and during the
            # forward it would delay kernel launches.
            if index+1<len(micros):ready=loader.submit(batch.prepare,index+1)
            else:pending=loader.submit(load_batch,outer+1) if outer+1<updates else None
            value.backward()
            for name,term in terms.items():
                term_sums[name]=term_sums.get(name,0)+term.detach()*(len(micro)/local_batch)
            del terms,value
        del batch
        terms=term_sums;value=sum(terms.values())*(world*local_batch/global_batch)
        grad_norm=average_gradients(parameters,world)
        grad_norm,totals=step_statistics(terms,value,grad_norm,local_batch,global_batch,world)
        used_rates=[g['lr'] for g in optimizer.param_groups]
        optimizer.step();scheduler.step();ema.update()
        if on_step(outer,grad_norm,dict(zip(terms,totals.tolist(),strict=True)),used_rates,drain):return outer+1
    return updates


def epoch_budget(cases,world,batch_per_rank,config,learning_rate,*,global_batch=None):
    """Steps per epoch and the schedule. An epoch is the number of global batches in which the sampler visits every
    state of the largest trial (with goal/no-goal strata alternating, its larger stratum times the strata count)
    while every trial gets the same share of rows."""
    import math
    batch=sum(rank_batch_sizes(world,batch_per_rank,global_batch))
    coverage=max(len(c[0]) if len(c)==1 else max(map(len,c[1:]))*(len(c)-1) for c in cases)
    steps=math.ceil(coverage*len(cases)/batch)
    return dict(checkpoint_every_epochs=config.checkpoint_every_epochs,steps_per_epoch=steps,epochs=config.epochs,warmup_epochs=config.warmup_epochs,
        total_steps=steps*config.epochs,warmup_steps=steps*config.warmup_epochs,
        minimum_factor=config.min_learning_rate_factor,peak_learning_rate=learning_rate,
        weight_decay=config.weight_decay,ema_decay=config.ema_decay)


def full_state(model,optimizer,scheduler,ema,rank_rng,steps,contract,raw_sha,ema_sha,warmstart):
    """The complete resume state of a checkpoint: AdamW, schedule, EMA, every rank's RNG."""
    state=dict(optimizer=optimizer.state_dict(),rank_rng=rank_rng,steps=steps,contract=contract,model_sha256=raw_sha)
    state.update(format=FULL_FORMAT,scheduler=scheduler.state_dict(),ema=ema.state_dict(),ema_model_sha256=ema_sha,warmstart=warmstart)
    return state


def restore_full(path,model,optimizer,scheduler,ema,*,contract,raw_sha,rank,device):
    """Load full_state into the optimizer, schedule, EMA and this rank's RNGs; refuses another contract or model."""
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    # The contract fixes the source code, data, batch, budget and GPU count; the raw weights must be the run's own.
    if state.get('contract')!=contract or state.get('model_sha256')!=raw_sha:
        raise ValueError(f'{path} belongs to another training contract or model; resume with the same run and raw weights')
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    ema.load_state_dict(state['ema'])
    rng=state['rank_rng'][rank];torch.set_rng_state(rng['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(rng['cuda'],device)
    np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    return state


def save_checkpoint(folder,actor,model,optimizer,scheduler,ema,contract,warmstart,base_config,steps,start,
                    batch_per_rank,budget,raw_parent_sha,resume_path,payloads,logs,source_root,*,create,validation_history=()):
    """Write folder/{model.pt (raw), ema_model.pt (EMA, the evaluated weights), training_state.pt, result.json}."""
    if create:folder.mkdir(parents=True,exist_ok=False)
    provenance=dict(actor.provenance,training_protocol=PROTOCOL,objective=OBJECTIVE,weights_kind='raw',
        candidates=CANDIDATES,integration_steps=INTEGRATION_STEPS,losses='CFM + TRAJ + GF + LH + LQ (docs/ARCHITECTURE.md section 3)',
        steps=steps,cumulative_updates=ema.num_updates,incremental_updates=steps-start,
        batch_per_rank=batch_per_rank,effective_batch=contract['effective_batch'],warmstart=warmstart,
        new_AdamW_stage=resume_path is None,new_EMA_stage=resume_path is None,parent_bundle_sha256=raw_parent_sha,
        optimizer_parent_state_sha256=sha(resume_path) if resume_path else None,epoch_budget=budget,
        learning_rates=[g['lr'] for g in optimizer.param_groups],EMA_num_updates=ema.num_updates,deterministic_algorithms=True,
        gradient_communication=contract['gradient_communication'],nccl_protocol=contract['nccl_protocol'],control_communication='nccl_cuda_step_totals_gloo_rng')
    raw_model=folder/'model.pt';save_bundle(raw_model,base_config=base_config,policy=actor.policy,head=actor.head,architecture=actor.architecture,provenance=provenance)
    ema_model=folder/'ema_model.pt'
    with ema.average_parameters(model):
        save_bundle(ema_model,base_config=base_config,policy=actor.policy,head=actor.head,architecture=actor.architecture,provenance=dict(provenance,weights_kind='ema'))
    state=full_state(model,optimizer,scheduler,ema,[p['rng'] for p in payloads],steps,contract,sha(raw_model),sha(ema_model),warmstart)
    state['validation_history']=list(validation_history)
    torch.save(state,folder/'training_state.pt')
    experiment_io.write(folder/'result.json',dict(validation_history=list(validation_history),complete=True,provenance=provenance,contract=contract,all_modules_trainable=all(p.requires_grad for p in model.parameters()),all_learning_rates_positive=True,
        training_rows=contract['row_counts']['all'],goal_rows=contract['row_counts']['goal'],no_goal_rows=contract['row_counts']['no_goal'],source_snapshot_root=str(source_root.resolve()),model_sha256=state['model_sha256'],ema_model_sha256=state['ema_model_sha256'],
        evaluation_model=str(ema_model.resolve()),evaluation_weights_kind='ema',training_state_sha256=sha(folder/'training_state.pt'),
        performance_by_rank=[p['performance'] for p in payloads],trace=logs))


# The one training run (docs/ARCHITECTURE.md): these defaults reproduce it with `bash scripts/local_planner/train.sh`.
DEFAULTS=dict(bundle=Path('data/model/initial.pt'),epochs=8,lr=1e-3,batch_per_rank=160,global_batch=2560,validation_every_epochs=1)


def configure_training_actor(actor,device):
    """Put the model in its training configuration on `device`: deterministic kernels, every module trainable (in
    eval mode: no dropout), bounded attention-bias activations, the training kernels and compiled graphs of
    execution.configure_training_execution, and compiled CNN, pixel encoder and geometry head. Returns the execution
    description with the attention budgets in bytes."""
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    for module in (actor.policy,actor.head):module.eval().requires_grad_(True)
    device_bytes=torch.cuda.get_device_properties(device).total_memory
    budgets=dict(generator=min(512<<20,device_bytes//(8*len(actor.policy.trajectory_decoder.blocks)*INTEGRATION_STEPS)),
        evaluator=min(8<<30,device_bytes//(6*len(actor.policy.trajectory_evaluator.blocks))))
    # Bound bias activations while leaving room for CNNs, attention and gradients.
    for block in actor.policy.trajectory_decoder.blocks:block.cross_attention.saved_bias_bytes=budgets['generator']
    for block in actor.policy.trajectory_evaluator.blocks:block.cross_attention.saved_bias_bytes=budgets['evaluator']
    from kappanav.local_planner.training.execution import configure_training_execution
    execution=configure_training_execution(actor)
    budgets['generator']=execution['generator_saved_bias_bytes']
    # Repeated CNN calls before one backward corrupt saved tensors with CUDA graphs on torch 2.7.
    encoder=actor.policy.depth_encoder
    encoder.image_features=torch.compile(encoder.image_features,fullgraph=True,dynamic=True,mode='default')
    encoder.encode_pixels=torch.compile(encoder.encode_pixels,fullgraph=True,dynamic=True,options={'emulate_precision_casts':True})
    actor.head.forward=torch.compile(actor.head.forward,fullgraph=True,dynamic=True,options={'emulate_precision_casts':True})
    return dict(execution,attention_activation_budgets_bytes=budgets)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,help=f"Initial model (default {DEFAULTS['bundle']}); with --resume, defaults to the checkpoint's raw model.pt")
    parser.add_argument('--dataset',type=Path,nargs='+',default=[Path('data/dataset')],help='Published dataset roots or dataset.json files, trained on together (default data/dataset)')
    parser.add_argument('--output',type=Path,help='New output directory (default outputs/training/<UTC time>)')
    parser.add_argument('--resume',type=Path,help='Complete checkpoint directory or training_state.pt of an interrupted run; restores AdamW, schedule, EMA and RNG')
    parser.add_argument('--epochs',type=int,help=f"Epoch/schedule budget (default {DEFAULTS['epochs']})")
    parser.add_argument('--warmup-epochs',type=int,help='Warmup epochs (default: bundle config), smaller than the epoch budget')
    parser.add_argument('--lr',type=float,help=f"Peak learning rate (default {DEFAULTS['lr']})")
    parser.add_argument('--seed',type=int,help='Sampling/model RNG seed (default: bundle config)')
    parser.add_argument('--batch-per-rank',type=int,help=f"Maximum samples per GPU forward/backward; larger global batches accumulate gradients (default {DEFAULTS['batch_per_rank']})")
    parser.add_argument('--global-batch',type=int,help=f"Effective batch across all visible GPUs; uneven partitions are supported (default {DEFAULTS['global_batch']})")
    parser.add_argument('--validation-every-epochs',type=int,help=f"Validation cadence (default {DEFAULTS['validation_every_epochs']})")
    args=parser.parse_args()
    external_world=int(os.environ.get('WORLD_SIZE','1'))
    external=external_world>1
    rank=int(os.environ['RANK']) if external else 0
    if args.output is None:
        if external:parser.error('--output is required under an external launcher')
        args.output=Path('outputs/training')/time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    for name in ('batch_per_rank','global_batch','validation_every_epochs','epochs'):
        if getattr(args,name) is not None and getattr(args,name)<1:parser.error(f"--{name.replace('_','-')} must be positive")
    if args.warmup_epochs is not None and args.warmup_epochs<0:parser.error('--warmup-epochs must be nonnegative')
    if rank==0 and args.output.exists():parser.error('Training output already exists; use a new output directory')
    proof=None
    if args.resume:
        args.resume=args.resume/'training_state.pt' if args.resume.is_dir() else args.resume
        proof=experiment_io.read(args.resume.with_name('result.json'))
        if not proof['complete'] or proof['training_state_sha256']!=sha(args.resume):parser.error('Resume state does not match a complete published checkpoint')
        if args.bundle is None:args.bundle=args.resume.with_name('model.pt')
        contract=proof['contract'];budget=contract['epoch_budget']
        for name,value in dict(validation_every_epochs=contract['validation_every_epochs'],batch_per_rank=contract['batch_per_rank'],
                               global_batch=contract['effective_batch'],epochs=budget['epochs'],warmup_epochs=budget['warmup_epochs'],
                               lr=budget['peak_learning_rate'],seed=contract['sampling_seed']).items():
            if getattr(args,name) is None:setattr(args,name,value)
    for name,value in DEFAULTS.items():
        if getattr(args,name) is None:setattr(args,name,value)
    args.bundle=(args.bundle/'model.pt' if args.bundle.is_dir() else args.bundle).resolve()
    if not args.bundle.is_file():parser.error(f'No model bundle at {args.bundle}')
    if proof and proof['model_sha256']!=sha(args.bundle):parser.error('Resume requires the checkpoint\'s matching raw model.pt, not EMA')
    saved=torch.load(args.bundle,map_location='cpu',weights_only=False,mmap=True)
    if saved.get('format')!=BUNDLE_FORMAT:parser.error('--bundle requires a model bundle; use --resume for training_state.pt')
    training=config_from_mapping(saved['base_config']).training
    if args.seed is None:args.seed=training.seed
    if args.warmup_epochs is None:args.warmup_epochs=training.warmup_epochs
    if not 0<=args.warmup_epochs<args.epochs:parser.error('--warmup-epochs must be smaller than --epochs; use --warmup-epochs 0 for a one-epoch run')
    if not np.isfinite(args.lr) or args.lr<=0:parser.error('Learning rate must be finite and positive')
    try:
        manifests,dataset_rows=dataset_manifests(args.dataset)
        args.manifests=manifests['train'];args.validation_manifests=manifests['validation']
    except (OSError,ValueError) as error:parser.error(str(error))
    if not args.manifests or dataset_rows['train']==0:parser.error('No completed training rows; finish the data pipeline or select a nonempty published dataset')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('NCCL_PROTO','LL')
    world=external_world if external else torch.cuda.device_count()
    if world<1:parser.error('No CUDA devices are visible; select available GPUs with CUDA_VISIBLE_DEVICES')
    if not 0<=args.seed<=np.iinfo(np.uint32).max-world+1:parser.error('--seed plus the GPU rank must fit the NumPy uint32 seed range')
    if proof and world!=proof['contract']['world_size']:parser.error('Full RNG restoration requires the original GPU count; use the same number of visible GPUs')
    if args.global_batch<world:parser.error('--global-batch must give each rank at least one sample')
    os.environ['KAPPANAV_SHARED_REPLAY_INPUTS']=json.dumps([str(args.bundle),list(map(str,args.manifests)),list(map(str,args.validation_manifests))])
    if external:
        train_worker(rank,world,args,None)
    else:
        mp.set_forkserver_preload(['kappanav.local_planner.training.shared_replay'])
        with tempfile.TemporaryDirectory(prefix='curvenav-train-') as folder:
            torch.multiprocessing.start_processes(train_worker,args=(world,args,Path(folder)/'rendezvous'),nprocs=world,start_method='forkserver')


def train_worker(rank,world,args,init_file):
    # The old generation is mostly live replay/compiler objects; keep young GC unchanged.
    young,middle,_=gc.get_threshold()
    gc.set_threshold(young,middle,100)
    from kappanav.local_planner.training.shared_replay import TRAINING, VALIDATION, SOURCE_KEYS, CASES
    items,goal_rows,no_goal_rows,data=TRAINING
    validation,_,_,validation_data=VALIDATION
    from kappanav.local_planner.training.optimizer import build_optimizer, build_cosine_schedule
    from kappanav.local_planner.training.ema import ExponentialMovingAverage
    local_rank=int(os.environ['LOCAL_RANK']) if init_file is None else rank
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(local_rank),WORLD_SIZE=str(world))
    global_batch=args.global_batch
    batch_sizes=rank_batch_sizes(world,args.batch_per_rank,global_batch)
    local_batch=batch_sizes[rank]
    microbatch_size=min(local_batch,args.batch_per_rank)
    torch.set_num_threads(1);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',torch.cuda.current_device())
    dist.init_process_group('nccl',init_method='env://' if init_file is None else init_file.as_uri(),rank=rank,world_size=world);control=dist.new_group(backend='gloo')
    actor=load_bundle(args.bundle,device);assert actor.policy.integration_steps==INTEGRATION_STEPS
    seed=args.seed+rank
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    actor.config=replace(actor.config,training=replace(actor.config.training,learning_rate=args.lr,
        epochs=args.epochs,warmup_epochs=args.warmup_epochs,seed=args.seed))
    execution=configure_training_actor(actor,device)
    model=torch.nn.ModuleDict(dict(policy=actor.policy,head=actor.head))
    dist.barrier()
    parameters=[p for p in model.parameters() if p.requires_grad]
    cases=CASES;source_keys=SOURCE_KEYS
    budget=epoch_budget(cases,world,args.batch_per_rank,actor.config.training,args.lr,global_batch=global_batch)
    optimizer=build_optimizer(model,args.lr,budget['weight_decay'])
    scheduler=build_cosine_schedule(optimizer,budget['total_steps'],budget['warmup_steps'],budget['minimum_factor'])
    ema=ExponentialMovingAverage(model,budget['ema_decay'])
    package=Path(__file__).resolve().parents[2]
    source_names=SOURCE_FILES
    # Everything a resume must reproduce: sources, data, batch partition, sampling, schedule and execution.
    contract=dict(training_protocol=PROTOCOL,objective=OBJECTIVE,sources={name:sha(package/name) for name in source_names},
        data=data,validation_data=validation_data,row_counts=dict(all=len(items),goal=len(goal_rows),no_goal=len(no_goal_rows)),
        world_size=world,effective_batch=global_batch,rank_batch_sizes=batch_sizes,batch_per_rank=args.batch_per_rank,
        validation_every_epochs=args.validation_every_epochs,validation_batch_sizes=[min(n,args.batch_per_rank) for n in batch_sizes],
        sampling=TRAIN_SAMPLING,sampling_seed=actor.config.training.seed,epoch_budget=budget,
        geometry_encoding_batch_size=actor.config.training.geometry_batch_size,
        actor_semantics=actor.architecture['actor_semantics'],precision=actor.architecture['precision'],
        integration_steps=INTEGRATION_STEPS,candidates=CANDIDATES,training_execution=execution,
        torch_version=torch.__version__,cuda_allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF','default'),
        gradient_communication='nccl_cuda_fp32_weighted_sample_mean',nccl_protocol=os.environ.get('NCCL_PROTO','default'))
    validation_history=[]
    start=0;raw_parent_sha=sha(args.bundle)
    warmstart=dict(bundle=str(args.bundle.resolve()),sha256=raw_parent_sha,previous_optimizer_restored=False)
    if args.resume:
        state=restore_full(args.resume,model,optimizer,scheduler,ema,contract=contract,raw_sha=raw_parent_sha,rank=rank,device=device)
        start=state['steps'];warmstart=state['warmstart'];validation_history=state['validation_history']
    updates=budget['total_steps']-start
    if updates<1:raise ValueError('The resumed run has already completed its schedule')
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=False)
        for name in source_names:
            p=args.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((package/name).read_bytes())
    base_config=config_to_mapping(actor.config)
    checkpoint_interval=budget['steps_per_epoch']*actor.config.training.checkpoint_every_epochs
    # Derived plan inputs of the batch in the forward and of the one being prepared stay cached.
    from kappanav.local_planner.features.history_fusion import reserve_plan_inputs
    reserve_plan_inputs(3*local_batch)
    dist.barrier();torch.cuda.reset_peak_memory_stats(device);logs=[];durations=[];began=time.monotonic()
    def load_batch(outer):
        ids=balanced_batch_indices(cases,outer,start,rank,world,args.batch_per_rank,source_keys,global_batch=global_batch,sampling_seed=actor.config.training.seed)
        return prepare_training_batch(actor,items,ids,bucket=execution['source_sampling_bucket'],microbatch_size=microbatch_size,cnn_batch_size=execution['cnn_batch_size'])
    def on_step(outer,grad_norm,totals,used_rates,drain):
        elapsed=time.monotonic()-step_began[0];durations.append(elapsed)
        if rank==0:
            record=dict(step=start+outer+1,seconds=time.monotonic()-began,
                step_seconds=elapsed,global_samples_per_second=global_batch/elapsed,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,learning_rates_used=used_rates,grad_norm=grad_norm,weights_kind='raw',aggregation='optimization_batch_pre_update',
                **totals);logs.append(record);print(json.dumps(record),flush=True)
        if (start+outer+1)%checkpoint_interval==0:
            drain()
            from kappanav.local_planner.training.convergence import comparison_losses
            monitor_began=time.monotonic()
            epoch=(start+outer+1)//budget['steps_per_epoch']
            validation_due=epoch%args.validation_every_epochs==0
            if validation and validation_due:
                with ema.average_parameters(model):
                    metrics=comparison_losses(actor,items,validation,rank,world,device,control,batch_size=microbatch_size)
                validation_history.append(dict(epoch=epoch,step=start+outer+1,
                    weights_kind='ema',aggregation=SAMPLING,train=metrics['train'],**metrics['validation']))
                if rank==0:print(json.dumps(dict(validation=validation_history[-1])),flush=True)
            torch.cuda.synchronize(device)
            monitoring_seconds=time.monotonic()-monitor_began
            checkpoint_began=time.monotonic()
            epoch_payloads=[None]*world
            stat=dict(rank=rank,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                device_total_bytes=torch.cuda.get_device_properties(device).total_memory,elapsed_seconds=time.monotonic()-began,
                local_samples=(outer+1)*local_batch,local_samples_per_second=(outer+1)*local_batch/(time.monotonic()-began))
            dist.all_gather_object(epoch_payloads,dict(rng=capture_rng(device),performance=stat),group=control)
            if rank==0:
                checkpoint=args.output/'checkpoints'/f"epoch_{(start+outer+1)//budget['steps_per_epoch']:03d}"
                save_checkpoint(checkpoint,actor,model,optimizer,scheduler,ema,contract,warmstart,base_config,start+outer+1,start,
                    args.batch_per_rank,budget,raw_parent_sha,args.resume,epoch_payloads,logs,args.output/'source',create=True,validation_history=validation_history)
            dist.barrier()
            if rank==0:print(json.dumps(dict(epoch_end=dict(epoch=(start+outer+1)//budget['steps_per_epoch'],step=start+outer+1,
                monitoring_seconds=monitoring_seconds,checkpoint_seconds=time.monotonic()-checkpoint_began))),flush=True)
        step_began[0]=time.monotonic()
        return False
    step_began=[time.monotonic()]
    with ThreadPoolExecutor(max_workers=1) as loader:
        updates=training_loop(actor,parameters,optimizer,scheduler,ema,load_batch,updates,local_batch=local_batch,
            microbatch_size=microbatch_size,world=world,global_batch=global_batch,cnn_batch_size=execution['cnn_batch_size'],
            loader=loader,on_step=on_step)
    torch.cuda.synchronize(device);elapsed=time.monotonic()-began
    perf=dict(rank=rank,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        device_total_bytes=torch.cuda.get_device_properties(device).total_memory,elapsed_seconds=elapsed,local_samples=updates*local_batch,
        local_samples_per_second=updates*local_batch/elapsed,outer_step_seconds=durations)
    payloads=[None]*world;dist.all_gather_object(payloads,dict(rng=capture_rng(device),performance=perf),group=control)
    if rank==0:
        save_checkpoint(args.output,actor,model,optimizer,scheduler,ema,contract,warmstart,base_config,start+updates,start,
            args.batch_per_rank,budget,raw_parent_sha,args.resume,payloads,logs,args.output/'source',create=False,validation_history=validation_history)
    dist.barrier();dist.destroy_process_group(control);dist.destroy_process_group()


if __name__=='__main__':main()
