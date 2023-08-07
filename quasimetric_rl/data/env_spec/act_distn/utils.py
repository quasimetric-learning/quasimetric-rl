from typing import *

import numpy as np

import torch
import torch.distributions
import torch.distributions.constraints


#-----------------------------------------------------------------------------#
#------------------------------ distributions --------------------------------#
#-----------------------------------------------------------------------------#


# https://github.com/deepmind/acme/blob/5fac34092330fbe7d758adc487c4b97c9f58f777/acme/jax/networks/distributional.py#L179
class AcmeTanhTransformedDistribution(torch.distributions.TransformedDistribution):
    def __init__(self, dist: torch.distributions.Distribution, threshold=.999, validate_args: bool = True):
        super().__init__(dist, torch.distributions.TanhTransform(), validate_args=validate_args)

        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = torch.atanh(torch.as_tensor(threshold))
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = np.log(1. - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.

        from .log_ndtr import log_ndtr_general
        assert isinstance(self.base_dist, torch.distributions.Normal)
        self._log_prob_left = log_ndtr_general(-inverse_threshold, self.base_dist.mean, self.base_dist.scale) - log_epsilon
        self._log_prob_right = log_ndtr_general(2 * self.base_dist.mean - inverse_threshold, self.base_dist.mean, self.base_dist.scale) - log_epsilon

    def log_prob(self, event):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = torch.clamp(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return torch.where(
            event <= -self._threshold,
            self._log_prob_left,
            torch.where(event >= self._threshold,
                        self._log_prob_right,
                        super().log_prob(event)))


# https://github.com/juliusfrost/dreamer-pytorch
class SampleDist(torch.distributions.Distribution):
    def __init__(self, dist: torch.distributions.Distribution, samples: int = 100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        sample = self._dist.rsample([self._samples])
        return torch.mean(sample, 0)

    @property
    def mode(self):
        sample: torch.Tensor = self._dist.rsample([self._samples])
        logprob: torch.Tensor = self._dist.log_prob(sample)
        assert len(self._dist.batch_shape) == 1, self._dist.batch_shape
        assert len(self._dist.event_shape) == 1
        assert logprob.ndim == 2
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        sample: torch.Tensor = self._dist.rsample([self._samples])
        logprob = self._dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()

    def rsample(self):
        return self._dist.rsample()

    def log_prob(self, value):
        return self._dist.log_prob(value)

    def __repr__(self):
        return f"SampleDist({self._dist})"