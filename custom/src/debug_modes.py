import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


def enable_nan_debug(config):
    torch.autograd.set_detect_anomaly(True)

    @HOOKS.register_module(force=True)
    class FirstNonFiniteHook(Hook):
        def _check(self, runner, kind, name, tensor):
            if torch.is_tensor(tensor) and tensor.numel() and not torch.isfinite(tensor).all():
                msg = f"[NaN] first non-finite {kind}: {name}"
                runner.logger.error(msg)
                raise RuntimeError(msg)

        def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            with torch.no_grad():
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        self._check(runner, "output", key, value)

                for name, param in runner.model.named_parameters():
                    self._check(runner, "param", name, param)

                for name, buffer in runner.model.named_buffers():
                    self._check(runner, "buffer", name, buffer)

                wrappers = getattr(runner, "optim_wrapper", None)
                if isinstance(wrappers, dict):
                    wrappers = list(wrappers.values())
                else:
                    wrappers = [wrappers]

                param_names = {id(param): name for name, param in runner.model.named_parameters()}

                for wrapper in wrappers:
                    if wrapper is None:
                        continue

                    optimizer = getattr(wrapper, "optimizer", None)
                    if optimizer is None:
                        continue

                    for param, state in list(getattr(optimizer, "state", {}).items()):
                        param_name = param_names.get(id(param), "unknown_param")
                        for state_key, state_value in list(state.items()):
                            self._check(runner, "optimizer_state", f"{param_name}.{state_key}", state_value)

    custom_hooks = list(config.get("custom_hooks", []) or [])
    custom_hooks.append(dict(type="FirstNonFiniteHook"))
    config.custom_hooks = custom_hooks
    return config
