"""Record the software versions that can affect experiment results."""

import importlib
import platform


PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "skimage",
    "matplotlib",
    "torch",
)


def collect_versions() -> dict[str, str]:
    """Return Python and library versions in a stable, printable mapping."""
    versions = {"python": platform.python_version()}
    for package_name in PACKAGES:
        module = importlib.import_module(package_name)
        versions[package_name] = module.__version__
    return versions
