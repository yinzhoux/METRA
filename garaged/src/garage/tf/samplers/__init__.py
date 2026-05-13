"""Samplers which run agents that use Tensorflow in environments."""
try:
    from garage.tf.samplers.batch_sampler import BatchSampler
    from garage.tf.samplers.worker import TFWorkerClassWrapper, TFWorkerWrapper
    __all__ = ['BatchSampler', 'TFWorkerClassWrapper', 'TFWorkerWrapper']
except ImportError:
    __all__ = []
