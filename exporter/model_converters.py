# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy
import torch

from rsl_rl.modules import EmpiricalNormalization


class ActorAndNormalizer(torch.nn.Module):
    """A class that collects `torch.nn.Sequential` and `EmpiricalNormalization` layers, as done by the Actor in rsl_rl."""

    def __init__(self, actor: torch.nn.Sequential, normalizer: EmpiricalNormalization = None, no_grad: bool = False):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer)
        self._no_grad = no_grad

    def in_features(self) -> int:
        return self.actor[0].in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._no_grad:
            x = x.detach()
        if self.normalizer:
            x = self.normalizer(x)
        return self.actor(x)


def recursive_to_nn_sequential(script_module: torch.jit.RecursiveScriptModule, no_grad: bool) -> torch.nn.Sequential:
    """
    Converts a `torch.jit.RecursiveScriptModule` back into a `torch.nn.Sequential` using raw torch.nn modules.
    """
    layers = []
    for child in script_module.children():
        if isinstance(child, torch.jit.RecursiveScriptModule):
            # Get the original type name
            original_type = child.original_name

            #  Map original types to nn.Module equivalents
            if original_type == "Linear":
                linear = torch.nn.Linear(
                    child.in_features,
                    child.out_features,
                    child.bias is not None,
                )
                linear.load_state_dict(child.state_dict())
                layers.append(linear)
            elif original_type == "ELU":
                elu = torch.nn.ELU(alpha=child.alpha)
                elu.load_state_dict(child.state_dict())
                layers.append(elu)
            elif original_type == "Tanh":
                tanh = torch.nn.Tanh()
                tanh.load_state_dict(child.state_dict())
                layers.append(tanh)
            elif original_type in ("Sequential", "MultiLayerPerceptron", "ModuleList"):
                # Recurse into inner Sequential
                layers.append(recursive_to_nn_sequential(child, no_grad=no_grad))
            else:
                raise NotImplementedError(f"Unsupported module: {original_type}")
        else:
            raise TypeError(f"Expected a `torch.jit.RecursiveScriptModule`, got: {type(child)}")
    network = torch.nn.Sequential(*layers)
    if no_grad:
        for param in network.parameters():
            param.requires_grad_(False)
        return network.eval()
    return network


def recursive_to_empirical_normalization(
    script_module: torch.jit.RecursiveScriptModule, no_grad: bool
) -> EmpiricalNormalization:
    """
    Converts a `torch.jit.RecursiveScriptModule` back into an `EmpiricalNormalization` using raw torch.nn modules.
    """
    # The shape passed to `EmpiricalNormalization` must exclude the batch dimension.
    shape = script_module._mean.shape[1:]
    normalizer = EmpiricalNormalization(shape=shape, eps=script_module.eps)
    normalizer.training = False
    normalizer.load_state_dict(script_module.state_dict())
    if no_grad:
        return normalizer.eval()
    return normalizer


def recursive_to_actor_and_normalizer(
    script_module: torch.jit.RecursiveScriptModule, no_grad: bool = False
) -> ActorAndNormalizer:
    """
    Converts a `torch.jit.RecursiveScriptModule` back into a `torch.nn.Sequential` and an `EmpiricalNormalization`
    using raw torch.nn modules.
    """
    actor = None
    normalizer = None

    for child in script_module.children():
        # Check if this is a Sequential or another module
        if isinstance(child, torch.jit.RecursiveScriptModule):
            # Get the original type name
            original_type = child.original_name

            if original_type in ("Sequential", "MultiLayerPerceptron"):
                actor = recursive_to_nn_sequential(script_module=child, no_grad=no_grad)
            elif original_type == "EmpiricalNormalization":
                normalizer = recursive_to_empirical_normalization(script_module=child, no_grad=no_grad)
            else:
                raise NotImplementedError(f"Unsupported module: {original_type}")
        else:
            raise TypeError("Unexpected module type")

    return ActorAndNormalizer(actor=actor, normalizer=normalizer, no_grad=no_grad)
