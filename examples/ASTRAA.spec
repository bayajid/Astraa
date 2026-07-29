# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('resources', 'resources')]
binaries = []
hiddenimports = ['PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'PyQt5.QtPrintSupport', 'PyQt5.QtNetwork', 'PyQt5.QtSvg', 'PyQt5.QtWebEngineWidgets', 'PyQt5.QtWebEngineCore', 'PyQt5.QtWebChannel', 'numpy', 'numpy.core', 'numpy.lib', 'numpy.fft', 'numpy.linalg', 'numpy.core._multiarray_tests', 'numpy.core._multiarray_umath', 'pandas', 'pandas.plotting', 'pandas._libs.tslibs.timedeltas', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_qt5agg', 'matplotlib.backends.backend_agg', 'scipy', 'scipy.interpolate', 'scipy.integrate', 'scipy.spatial.transform', 'scipy.spatial.transform._rotation_groups', 'scipy.special._ufuncs_cxx', 'scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack', 'astropy', 'astropy.coordinates', 'astropy.time', 'astropy.units', 'astropy.wcs', 'astropy.io.fits', 'astropy.table', 'skyfield', 'skyfield.api', 'skyfield.almanac', 'sgp4', 'sgp4.api', 'plotly', 'plotly.graph_objs', 'plotly.express', 'cartopy', 'cartopy.crs', 'cartopy.feature', 'tqdm', 'termcolor', 'requests', 'pytz', 'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont', 'pkg_resources', 'pkg_resources.extern']
tmp_ret = collect_all('PIL')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pandas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('setuptools')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pkg_resources')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('astropy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('skyfield')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cartopy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['/home/bkhan/Documents/Git/astropynaric/examples/astraa.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'turtle', 'PyQt6', 'PySide2', 'PySide6', 'charset_normalizer.md__mypyc'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ASTRAA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
