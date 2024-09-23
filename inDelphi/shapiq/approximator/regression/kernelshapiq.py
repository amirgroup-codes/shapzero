"""Regression with Shapley interaction index (SII) approximation."""

from typing import Optional

import numpy as np

from ._base import Regression

AVAILABLE_INDICES_KERNELSHAPIQ = {"k-SII", "SII"}


class KernelSHAPIQ(Regression):
    """Estimates the SII values using KernelSHAP-IQ.

    Algorithm described in `Fumagalli et al. (2024) <https://doi.org/10.48550/arXiv.2405.10852>`_.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation. Defaults to ``2``.
        index: The interaction index to be used. Choose from ``['k-SII', 'SII']``. Defaults to
            ``'k-SII'``.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        random_state: The random state of the estimator. Defaults to ``None``.
    """

    def __init__(
        self,
        n: int,
        max_order = 2,
        index = "k-SII",
        pairing_trick = False,
        sampling_weights = None,
        random_state = None,
    ) -> None:
        if index not in AVAILABLE_INDICES_KERNELSHAPIQ:
            raise ValueError(
                "Index {} not available for KernelSHAP-IQ. Choose from {}.".format(
                    index,
                    AVAILABLE_INDICES_KERNELSHAPIQ
                )
            )
        super().__init__(
            n,
            max_order,
            index=index,
            sii_consistent=True,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )


class InconsistentKernelSHAPIQ(Regression):
    """Estimates the SII values using Inconsistent KernelSHAP-IQ. Algorithm similar to kADD-SHAP.
    For details, refer to `Fumagalli et al. (2024) <https://doi.org/10.48550/arXiv.2405.10852>`_.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation. Defaults to ``2``.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        random_state: The random state of the estimator. Defaults to ``None``.
    """

    def __init__(
        self,
        n,
        max_order = 2,
        index = "k-SII",
        pairing_trick = False,
        sampling_weights = None,
        random_state = None,
    ) -> None:
        if index not in AVAILABLE_INDICES_KERNELSHAPIQ:
            raise ValueError(
                "Index {} not available for KernelSHAP-IQ. Choose from {}.".format(
                    index,
                    AVAILABLE_INDICES_KERNELSHAPIQ
                )
            )
        super().__init__(
            n,
            max_order,
            index=index,
            sii_consistent=False,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
