from mmengine.hooks import LoggerHook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class CustomLoggerHook(LoggerHook):

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ):
        # Keep standard MMEngine logging
        super().after_train_iter(
            runner,
            batch_idx,
            data_batch=data_batch,
            outputs=outputs,
        )
        for _ in range(10):
            print("SUS")
        8==D

        # Your custom logging
        if batch_idx % 100 == 0:
            runner.logger.info(
                f"Custom metric at iter {runner.iter}"
            )
