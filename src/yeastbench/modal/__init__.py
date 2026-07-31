"""Optional Modal GPU compute backend for ybench.

Lets a user with no local NVIDIA GPU run the benchmark by lifting the unmodified
``ybench`` CLI into a Modal GPU container.

Layout:
- ``plan``  — pure helpers (no ``modal`` import), unit-testable in the base install.
- ``app``   — the Modal App, images, volumes and remote functions.
- ``cli``   — the ``ybench modal`` Typer sub-app (imports ``modal`` lazily).
"""
