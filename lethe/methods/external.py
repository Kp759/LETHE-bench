"""Bringing your own method.

    --method checkpoint --method-arg path=/models/my_unlearned
        a model unlearned elsewhere (ROME / MEMIT / ZeroUnlearn runs, or a submission).
        The checkpoint must have been produced from the split's `requests` only.
    --method plugin --method-arg target=my_package.my_module:MyMethod
        any class implementing lethe.methods.base.Method.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import torch

from ..model_api import ModelHandle
from .base import Method, MethodCard, arg


class Checkpoint(Method):
    card = MethodCard("checkpoint", "external", "an HF checkpoint unlearned outside this harness")

    def unlearn(self, base, requests, allowed_retain, ctx):
        from transformers import AutoModelForCausalLM
        path = self.args["path"]
        dtype = next(base.hf.parameters()).dtype
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(base.device).eval()
        self.card = MethodCard(self.args.get("name", "checkpoint"), self.args.get("family", "external"),
                               f"checkpoint {path}")
        return ModelHandle(model=m, tokenizer=base.tokenizer, hf=m)


class Plugin(Method):
    card = MethodCard("plugin", "external", "user-supplied Method class")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mod, cls = str(kwargs["target"]).split(":")
        impl = getattr(importlib.import_module(mod), cls)
        rest = {k: v for k, v in kwargs.items() if k != "target"}
        self.impl = impl(**rest)
        self.card = self.impl.card

    def unlearn(self, base, requests, allowed_retain, ctx):
        return self.impl.unlearn(base, requests, allowed_retain, ctx)

    def unlearn_sequential(self, current, new_requests, history, allowed_retain, ctx):
        return self.impl.unlearn_sequential(current, new_requests, history, allowed_retain, ctx)

    def report(self):
        return self.impl.report()
