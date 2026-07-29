"""
make_astraa_pkg.py
==================
Build script for the astraa package.

Place this file inside  examples/  (which is where it already lives).
Run it from anywhere — all paths are resolved relative to the script itself.

Usage
-----
  python examples/make_astraa_pkg.py --mode user   # Standalone binary, no source exposed
  python examples/make_astraa_pkg.py --mode dev    # Editable install in current env

tudatpy note
------------
tudatpy is NOT on PyPI. If missing it is installed via conda into the
currently active environment's prefix. Nothing is ever created or switched.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Paths — everything anchored to the script's own location
# ---------------------------------------------------------------------------

HERE      = os.path.dirname(os.path.abspath(__file__))   # .../examples/
REPO_ROOT = os.path.dirname(HERE)                         # .../astropynaric/

PKG_NAME   = "astraa_pkg"
ENTRY_POINT = os.path.join(HERE, "astraa.py")             # examples/astraa.py
INPUT_DATA  = os.path.join(HERE, "input_data")            # examples/input_data/
PKG_ROOT    = os.path.join(HERE, PKG_NAME)                # examples/astraa_pkg/
DIST_DIR    = os.path.join(HERE, "dist")
BUILD_DIR   = os.path.join(HERE, "build")
SETUP_PY    = os.path.join(HERE, "setup.py")
REQ_TXT     = os.path.join(HERE, "requirements.txt")

# Modules live at the repo root, copied into the package under the same name
MODULE_NAMES = [
    "attitude_tools",
    "basic_tools",
    "pointing_calculations",
    "prediction_methods",
    "tudat_tools",
    "astronomy_tools",
    "plotting_tools",
    "gs_sc_tools",
    "link_processing_tools",
    "paa_tools",
    os.path.join("analyses", "attitude_predictions"),
]

# ---------------------------------------------------------------------------
# Dependency lists
# ---------------------------------------------------------------------------

# import name -> pip package name
PYPI_DEPS = {
    "numpy":      "numpy",
    "pandas":     "pandas",
    "matplotlib": "matplotlib",
    "plotly":     "plotly",
    "scipy":      "scipy",
    "skyfield":   "skyfield",
    "astropy":    "astropy",
    "pytz":       "pytz",
    "requests":   "requests",
}

# import name -> conda package name  (must NOT go through pip)
# PyQt5 must come from conda — the pip wheel is a stub that omits Qt5 plugin
# binaries, causing PyInstaller to fail with "plugins directory does not exist".
# cartopy also has C extensions that are more reliable from conda-forge.
CONDA_DEPS = {
    "tudatpy": "tudatpy",
    "PyQt5":   "pyqt",      # conda package is 'pyqt', not 'PyQt5'
    "cartopy": "cartopy",
}

CONDA_CHANNELS = ["-c", "tudat-team", "-c", "conda-forge"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    print(f"\n  >> {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def is_importable(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def find_conda() -> str | None:
    for exe in ("mamba", "conda"):
        path = shutil.which(exe)
        if path:
            return path
    return None

# ---------------------------------------------------------------------------
# Step 1 – Check / install deps into the CURRENT environment only
# ---------------------------------------------------------------------------

def pyqt5_plugins_exist() -> bool:
    """Return True if the Qt5 plugin directory actually exists (conda install).
    The pip PyQt5 wheel is a stub — it has no Qt5/plugins dir, which causes
    PyInstaller to crash.  If the dir is missing we must replace the pip stub
    with the full conda package before building.
    """
    try:
        import PyQt5
        plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
        return os.path.isdir(plugins)
    except ImportError:
        return False


def ensure_pyqt5_from_conda(conda: str):
    """Remove any pip-installed PyQt5 stub and install the full conda package."""
    # Check if pip owns a PyQt5 entry
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "PyQt5"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        # pip stub present — uninstall it so conda version wins
        print("[deps] Removing pip-installed PyQt5 stub …")
        run([sys.executable, "-m", "pip", "uninstall", "-y", "PyQt5", "PyQt5-sip", "PyQt5-Qt5"])

    print("[deps] Installing full PyQt5 via conda …")
    run([conda, "install", "--yes", "-p", sys.prefix, *CONDA_CHANNELS, "pyqt"])


def ensure_deps():
    print(f"\n[env] Active Python : {sys.executable}")
    print(f"[env] Active prefix : {sys.prefix}\n")

    conda = find_conda()

    # --- PyQt5: must come from conda (pip wheel has no Qt5 plugin binaries) ---
    if not pyqt5_plugins_exist():
        if not conda:
            sys.exit(
                "ERROR: PyQt5 Qt5/plugins directory not found.\n"
                "The pip wheel of PyQt5 is a stub — it must be replaced with the\n"
                "conda package (conda install pyqt).  conda/mamba was not found on PATH.\n"
                "Install Miniconda, activate your env, then re-run."
            )
        ensure_pyqt5_from_conda(conda)
    else:
        print("[deps] PyQt5 (conda/full) already present ✓")

    # --- other conda-only packages ---
    other_conda = {imp: pkg for imp, pkg in CONDA_DEPS.items() if imp != "PyQt5"}
    missing_conda = [pkg for imp, pkg in other_conda.items() if not is_importable(imp)]
    if missing_conda:
        if not conda:
            sys.exit(
                "ERROR: Missing conda-only packages: " + str(missing_conda) + "\n"
                "conda/mamba not found on PATH."
            )
        print(f"[deps] conda installing: {missing_conda}")
        run([conda, "install", "--yes", "-p", sys.prefix, *CONDA_CHANNELS, *missing_conda])
    else:
        print("[deps] Other conda packages already present ✓")

    # --- PyPI packages ---
    missing_pip = [pip for imp, pip in PYPI_DEPS.items() if not is_importable(imp)]
    if missing_pip:
        print(f"[deps] pip installing: {missing_pip}")
        run([sys.executable, "-m", "pip", "install", *missing_pip])
    else:
        print("[deps] PyPI packages already present ✓")

    # --- PyInstaller ---
    if not is_importable("PyInstaller"):
        print("[deps] Installing PyInstaller …")
        run([sys.executable, "-m", "pip", "install", "pyinstaller"])

# ---------------------------------------------------------------------------
# Step 2 – Build source tree (shared)
# ---------------------------------------------------------------------------

def build_source_tree():
    print(f"\n[pkg] Building source tree in '{PKG_ROOT}/' …")
    os.makedirs(PKG_ROOT, exist_ok=True)

    # Entry-point script
    shutil.copy(ENTRY_POINT, os.path.join(PKG_ROOT, "astraa.py"))

    # Source modules from repo root
    for rel_name in MODULE_NAMES:
        src = os.path.join(REPO_ROOT, rel_name)
        dst = os.path.join(PKG_ROOT, rel_name)
        if not os.path.exists(src):
            print(f"  [warn] Not found, skipping: {src}")
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    # __init__.py in every sub-directory
    for root, _dirs, _files in os.walk(PKG_ROOT):
        init = os.path.join(root, "__init__.py")
        if not os.path.exists(init):
            open(init, "a").close()

    # Optional input data
    if os.path.exists(INPUT_DATA):
        shutil.copytree(INPUT_DATA, os.path.join(PKG_ROOT, "input_data"), dirs_exist_ok=True)

# ---------------------------------------------------------------------------
# Step 3a – Developer mode
# ---------------------------------------------------------------------------

DEV_SETUP_PY = textwrap.dedent("""\
    from setuptools import setup, find_packages

    setup(
        name="{pkg_name}",
        version="0.1.0",
        description="astraa — developer install",
        packages=find_packages(),
        # tudatpy is conda-only — deliberately excluded from install_requires
        install_requires={pypi_deps!r},
        include_package_data=True,
        python_requires=">=3.9",
    )
""")

def build_dev():
    build_source_tree()

    pypi_list = list(PYPI_DEPS.values())

    with open(SETUP_PY, "w") as f:
        f.write(DEV_SETUP_PY.format(pkg_name=PKG_NAME, pypi_deps=pypi_list))

    with open(REQ_TXT, "w") as f:
        f.write("# tudatpy must be installed via conda (not pip):\n")
        f.write("#   conda install -c tudat-team tudatpy\n\n")
        f.write("\n".join(pypi_list) + "\n")

    print("\n[dev] Installing astraa_pkg as editable in the current env …")
    run([sys.executable, "-m", "pip", "install", "-e", HERE])

    print("\n✅  Developer setup complete.")
    print(f"   Run:  python -m {PKG_NAME}.astraa")

# ---------------------------------------------------------------------------
# Step 3b – User / binary mode
# ---------------------------------------------------------------------------

def build_user():
    build_source_tree()

    entry = os.path.join(PKG_ROOT, "astraa.py")

    print("\n[binary] Running PyInstaller in current env …")
    run([
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name",     "astraa",
        "--distpath", DIST_DIR,
        "--workpath", BUILD_DIR,
        "--noconfirm",
        "--clean",
        "--collect-data",  "tudatpy",
        "--collect-all",   "PyQt5",
        "--hidden-import", "tudatpy",
        "--hidden-import", "astropy",
        "--hidden-import", "skyfield",
        "--hidden-import", "cartopy",
        # Exclude every competing Qt binding so PyInstaller never runs their hooks
        "--exclude-module", "PySide6",
        "--exclude-module", "PySide2",
        "--exclude-module", "PyQt6",
        "--exclude-module", "PyQt4",
        entry,
    ])

    # Remove intermediate source — users never see it
    shutil.rmtree(PKG_ROOT,   ignore_errors=True)
    shutil.rmtree(BUILD_DIR,  ignore_errors=True)
    spec = os.path.join(HERE, "astraa.spec")
    if os.path.exists(spec):
        os.remove(spec)

    binary = os.path.join(DIST_DIR, "astraa")
    if sys.platform.startswith("win"):
        binary += ".exe"

    print(f"\n✅  User binary created: {binary}")
    print( "   Ship only that single file — no Python or source required.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build astraa using whichever Python environment is currently active."
    )
    parser.add_argument(
        "--mode", choices=["user", "dev"], required=True,
        help="user = standalone binary; dev = editable install in current env",
    )
    args = parser.parse_args()

    ensure_deps()

    if args.mode == "dev":
        build_dev()
    else:
        build_user()


if __name__ == "__main__":
    main()