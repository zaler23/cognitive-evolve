from __future__ import annotations

import importlib
import pkgutil

import cognitive_evolve_runtime


def test_every_runtime_module_imports() -> None:
    modules = pkgutil.walk_packages(
        cognitive_evolve_runtime.__path__,
        cognitive_evolve_runtime.__name__ + ".",
    )

    for module in modules:
        importlib.import_module(module.name)
