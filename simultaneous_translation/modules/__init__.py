# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from fairseq import registry
(
    build_monotonic_attention,
    register_monotonic_attention,
    MONOTONIC_ATTENTION_REGISTRY
) = registry.setup_registry('--simul-type')

from .monotonic_multihead_attention import MonotonicAttention, MonotonicMultiheadAttentionHard, MonotonicMultiheadAttentionInfiniteLookback, MonotonicMultiheadAttentionWaitk
from .monotonic_transformer_layer import TransformerMonotonicEncoderLayer, TransformerMonotonicDecoderLayer

