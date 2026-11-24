"""Training batches: the records of one optimization step and their microbatches' pinned host inputs, built on the
loader thread."""
from kappanav.local_planner.features.history_fusion import (_plan_inputs, memory_raster_inputs, source_fusion_inputs,
                                                           source_groups, source_sample_inputs, static_splat_segments)
from kappanav.local_planner.features.staging import HostTree
from kappanav.local_planner.training.bspline_objective import microbatch_inputs


class MicroBatch(list):
    """Records of one forward and their host inputs, packed for one pinned transfer per dtype."""
    def __init__(self,records,host=None):
        super().__init__(records);self.host=host


class TrainingBatch(list):
    """One optimization step's records, split into microbatches whose host inputs are built one at a time
    (prepare(index)), so the loader can build microbatch i+1 while microbatch i backpropagates."""
    def __init__(self,records,microbatch_size,actor=None,*,bucket=None,cnn_batch_size=None):
        super().__init__(records);size=microbatch_size or max(1,len(records))
        self.parts=[MicroBatch(records[offset:offset+size]) for offset in range(0,len(records),size)]
        self.actor,self.bucket,self.cnn_batch_size=actor,bucket,cnn_batch_size

    def microbatches(self):
        return self.parts

    def prepare(self,index):
        """Build and pin microbatch `index`'s host inputs (no-op without cnn_batch_size or past the end)."""
        if self.cnn_batch_size is None or index>=len(self.parts):return
        micro=self.parts[index];prepare_plans(self.actor,micro,bucket=self.bucket)
        micro.host=HostTree(microbatch_inputs(self.actor,micro,cnn_batch_size=self.cnn_batch_size,bucket=self.bucket))


def prepare_plans(actor,records,*,bucket=None):
    """Immutable per-plan CPU calibration (pixel sampling, fusion, raster and splat inputs) of these records."""
    plans={}
    for item,raw,*_ in records:plans[id(item['history_plan'])]=(item['history_plan'],raw)
    for plan,raw in plans.values():
        groups=source_groups(plan)
        if len(plan['lengths']):
            height,width=raw['depths'].shape[-2:]
            # The backbone and calibrated pixel sampler both use a stride-16 lattice.
            shape=(1,actor.policy.depth_encoder.spatial_projection.out_channels,(height+15)//16,(width+15)//16)
            source_sample_inputs(plan,{int(i):shape for i,_ in groups},{int(i):width for i,_ in groups},bucket=bucket)
            source_fusion_inputs(actor.policy.depth_encoder,plan)
        values=memory_raster_inputs(actor.policy,actor.head,plan)
        cache=_plan_inputs(plan).setdefault('_splat_segments',{})
        key=(actor.head.horizon,actor.head.raster_padding)
        if key not in cache:cache[key]=static_splat_segments(values[-1])


def prepare_training_batch(actor,items,indices,*,bucket=None,microbatch_size=None,cnn_batch_size=None):
    """Prepare immutable CPU calibration while the previous batch trains.

    With cnn_batch_size, the work is per microbatch (TrainingBatch.prepare: its plans, then its image chunks and
    fusion/raster/critic/target tensors packed into pinned buffers); only the first is built here, and the training
    loop prepares each next one during the current backward. Host preparation draws no random numbers, so when it
    runs does not change any value.
    """
    records=[items[index] for index in indices]
    batch=TrainingBatch(records,microbatch_size,actor,bucket=bucket,cnn_batch_size=cnn_batch_size)
    if cnn_batch_size is None:prepare_plans(actor,records,bucket=bucket)
    batch.prepare(0)
    return batch
