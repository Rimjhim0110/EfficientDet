import math
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class WarmupCosineDecayLRScheduler(LearningRateSchedule):
    def __init__(self, 
                 max_lr: float,
                 warmup_steps: int,
                 decay_steps: int,
                 alpha: float = 0.) -> None:
        super(WarmupCosineDecayLRScheduler, self).__init__()
        self.name = 'WarmupCosineDecayLRScheduler'
        self.alpha = alpha
        self.max_lr = max_lr
        self.last_step = 0
        self.warmup_steps = int(warmup_steps)
        self.linear_increase = self.max_lr / float(self.warmup_steps)
        self.decay_steps = int(decay_steps)
        #The constructor initializes the WarmupCosineDecay learning rate scheduler with maximum learning rate, warmup steps, decay steps, and an optional alpha parameter.

    def _decay(self) -> tf.Tensor:
        rate = tf.subtract(self.last_step, self.warmup_steps) 
        rate = tf.divide(rate, self.decay_steps)
        rate = tf.cast(rate, tf.float32)
        cosine_decayed = tf.multiply(tf.constant(math.pi), rate)
        cosine_decayed = tf.add(1., tf.cos(cosine_decayed))
        cosine_decayed = tf.multiply(.5, cosine_decayed)
        decayed = tf.subtract(1., self.alpha)
        decayed = tf.multiply(decayed, cosine_decayed)
        decayed = tf.add(decayed, self.alpha)
        return tf.multiply(self.max_lr, decayed)
      # Computes the cosine decay rate for the learning rate schedule.

    @property
    def current_lr(self) -> tf.Tensor:
        return tf.cond(
            tf.less(self.last_step, self.warmup_steps),
            lambda: tf.multiply(self.linear_increase, self.last_step),
            lambda: self._decay())
# Returns the current learning rate based on whether the step is in the warm-up phase or the decay phase.
    def __call__(self, step: int) -> tf.Tensor:
        self.last_step = step
        return self.current_lr
#Updates the internal state and returns the current learning rate for a given step.
    def get_config(self) -> dict:
        config = {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            'decay_steps': self.decay_steps,
            'alpha': self.alpha
        }
        return config
# Returns a dictionary containing the configuration of the learning rate scheduler, including max learning rate, warmup steps, decay steps, and alpha.
