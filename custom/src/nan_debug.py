import os

import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

_MAX_DEPTH = 6
_EVERY_FIRST_ITERS = 50
_EVERY_N_ITERS_AFTER = 25


def _iter_tensors(value, name="", depth=0):
    if depth > _MAX_DEPTH:
        return

    if torch.is_tensor(value):
        yield name, value
        return

    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    elif hasattr(value, "keys") and hasattr(value, "__getitem__"):
        items = ((key, value[key]) for key in value.keys())
    else:
        return

    for key, child in items:
        child_name = f"{name}.{key}" if name else str(key)
        yield from _iter_tensors(child, child_name, depth + 1)


def _resolve_interval(interval):
    if interval is not None:
        return max(1, int(interval))

    from_env = os.getenv("DEBUG_NAN_EVERY")
    if from_env:
        return max(1, int(from_env))

    return _EVERY_N_ITERS_AFTER


def nan_debug(config, interval=None, dense_iters=None):
    default_interval = _resolve_interval(interval)
    default_dense_iters = _EVERY_FIRST_ITERS if dense_iters is None else max(0, int(dense_iters))

    torch.autograd.set_detect_anomaly(True)

    @HOOKS.register_module(force=True)
    class FirstNonFiniteHook(Hook):
        def __init__(self, interval=default_interval, dense_iters=default_dense_iters):
            super().__init__()
            self.interval = max(1, int(interval))
            self.dense_iters = max(0, int(dense_iters))
            self.step = 0

        def _check(self, runner, kind, name, tensor):
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                return
            if not tensor.numel() or torch.isfinite(tensor).all():
                return

            sampled = self.step > self.dense_iters and self.interval > 1
            lag = "" if not sampled else f" (may be up to {self.interval - 1} iters late)"
            msg = f"[NaN] non-finite {kind}: {name} (checked at iter {self.step}){lag}"
            runner.logger.error(msg)
            print(msg, flush=True)
            raise RuntimeError(msg)

        def _check_outputs(self, runner, outputs):
            for name, tensor in _iter_tensors(outputs, "outputs"):
                self._check(runner, "output", name, tensor)

        def _check_params(self, runner):
            for name, param in runner.model.named_parameters():
                self._check(runner, "param", name, param)

            for name, buffer in runner.model.named_buffers():
                self._check(runner, "buffer", name, buffer)

        def _check_optimizer_state(self, runner):
            wrappers = getattr(runner, "optim_wrapper", None)
            if isinstance(wrappers, dict):
                wrappers = list(wrappers.values())
            elif wrappers is None:
                wrappers = []
            else:
                wrappers = [wrappers]

            param_names = {id(param): name for name, param in runner.model.named_parameters()}

            for wrapper in wrappers:
                optimizer = getattr(wrapper, "optimizer", None)
                if optimizer is None:
                    continue

                for param, state in list(getattr(optimizer, "state", {}).items()):
                    param_name = param_names.get(id(param), "unknown_param")
                    for state_key, state_value in list(state.items()):
                        self._check(runner, "optimizer_state", f"{param_name}.{state_key}", state_value)

        def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            step = getattr(runner, "iter", None)
            if not isinstance(step, int):
                step = batch_idx + 1
            self.step = step

            step_interval = 1 if step <= self.dense_iters else self.interval
            if step_interval > 1 and step % step_interval != 0:
                return

            with torch.no_grad():
                self._check_outputs(runner, outputs)
                self._check_params(runner)
                self._check_optimizer_state(runner)

    custom_hooks = list(config.get("custom_hooks", []) or [])
    custom_hooks.append(dict(type="FirstNonFiniteHook", interval=default_interval, dense_iters=default_dense_iters))
    config.custom_hooks = custom_hooks
    return config
