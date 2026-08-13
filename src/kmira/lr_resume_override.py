"""Let a resumed run use NEW warmup/constant/decay/min_lr values instead of the checkpointed ones.

``WarmupConstantCosineDecayLR`` doesn't override ``state_dict``/``load_state_dict``, so it inherits
torch's ``LRScheduler`` default: dump/restore the *entire* ``__dict__``, config included. mira's
``CheckpointManager.continue_from`` calls that on every registered component, so a resumed run's
``warmup_steps``/``constant_steps``/``decay_steps``/``min_lr`` are silently overwritten by whatever
was saved in the checkpoint -- regardless of what this invocation's Hydra config asked for.

That is invisible right up until it matters: annealing a checkpoint trained at constant LR
(``decay_steps=0``) means passing ``decay_steps>0`` on resume, which would construct the scheduler
correctly, run with no error, and then have its constructor values immediately clobbered back to
``decay_steps=0`` -- an anneal that silently doesn't anneal.

This patches ``load_state_dict`` to keep this run's own schedule-defining attributes, restoring
everything else -- crucially ``last_epoch``, the step counter the LR is actually a function of --
as normal, so resume position is still exact.
"""

from __future__ import annotations

from mira.training.lr_schedule import WarmupConstantCosineDecayLR

_SCHEDULE_CONFIG_KEYS = frozenset({"warmup_steps", "constant_steps", "decay_steps", "min_lr"})

_original_load_state_dict = WarmupConstantCosineDecayLR.load_state_dict


def _load_state_dict_keep_new_schedule(self, state_dict: dict) -> None:
    filtered = {k: v for k, v in state_dict.items() if k not in _SCHEDULE_CONFIG_KEYS}
    self.__dict__.update(filtered)


def use_new_schedule_on_resume() -> None:
    """Patch ``WarmupConstantCosineDecayLR`` in this process. Idempotent."""
    if WarmupConstantCosineDecayLR.load_state_dict is not _load_state_dict_keep_new_schedule:
        WarmupConstantCosineDecayLR.load_state_dict = _load_state_dict_keep_new_schedule
