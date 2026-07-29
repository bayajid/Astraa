#!/usr/bin/env python3
"""
ASTRAA Build Script - WITH PROGRESS INDICATOR
Shows real-time build progress
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import platform
import threading
import time

VERSION = "1.0.0"
MAIN_FILE = "astraa.py"

class ASTRAABuilder:
    def __init__(self):
        self.root = Path.cwd()
        self.dist_dir = self.root / 'releases'
        self.build_date = datetime.now().strftime("%Y%m%d_%H%M")
        self.system = platform.system()
        self.is_windows = self.system == 'Windows'
        self.is_macos = self.system == 'Darwin'
        self.is_linux = self.system == 'Linux'
        self.building = False
    
    def show_progress_spinner(self):
        """Show a spinner while building"""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        idx = 0
        start_time = time.time()
        
        while self.building:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            
            # Show spinner with elapsed time
            sys.stdout.write(f'\r{spinner[idx]} Building... [{mins:02d}:{secs:02d}] ')
            sys.stdout.flush()
            
            idx = (idx + 1) % len(spinner)
            time.sleep(0.1)
        
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * 50 + '\r')
        sys.stdout.flush()
    
    def check_tudat(self):
        """Check if tudatpy is installed"""
        try:
            import tudatpy
            tudat_path = Path(tudatpy.__file__).parent
            print(f"✓ Tudatpy found at: {tudat_path}")
            return tudat_path
        except ImportError:
            print("⚠️  WARNING: tudatpy not found!")
            print("   Tudat features will NOT be available in the binary")
            print("   To include tudat, install it first:")
            print("   conda install -c tudat-team tudatpy")
            return None

    def get_tudat_binaries(self, tudat_path):
        """Find tudat binary files and build correct --add-binary args.
        
        tudatpy.kernel is a compiled C extension. PyInstaller's collect-all
        alone is not enough — we must explicitly add every .so with the correct
        destination path so Python's import system can find them inside the bundle.
        """
        if not tudat_path:
            return []

        site_packages = tudat_path.parent  # .../site-packages
        args = []

        # --paths: tells PyInstaller where to search for imports at analysis time
        args.append(f'--paths={site_packages}')

        # --add-binary: copies each .so into the correct subdir inside the bundle
        # dest must be relative to bundle root and match the Python module path
        # e.g. tudatpy/kernel/_core.so → importable as tudatpy.kernel._core
        binaries_found = 0
        for ext in ['*.so*', '*.pyd', '*.dll', '*.dylib']:
            for file in tudat_path.rglob(ext):
                try:
                    dest_dir = file.parent.relative_to(site_packages)
                    args.append(f'--add-binary={file}{os.pathsep}{dest_dir}')
                    binaries_found += 1
                except ValueError:
                    pass  # skip files outside site-packages

        print(f"  Found {binaries_found} tudat binary files")
        return args
    
    def build_binary(self, main_file):
        """Build with real-time progress"""
        print(f"🔨 Building binary from: {main_file}")
        print(f"🖥️  Platform: {self.system}")
        
        # Check for tudatpy
        tudat_path = self.check_tudat()
        tudat_binaries = self.get_tudat_binaries(tudat_path) if tudat_path else []
        
        # Find data folders
        print("📁 Scanning for data folders...")
        data_args = self.find_data_folders()
        
        # PyInstaller command
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--name=ASTRAA',
            '--onefile',
            '--clean',
            '--noconfirm',
        ]

        # --windowed suppresses the terminal window on macOS/Windows (GUI app).
        # On Linux it hides ALL crash output making debugging impossible — skip it.
        if not self.is_linux:
            cmd.append('--windowed')
        
        cmd.extend(data_args)
        
        if tudat_binaries:
            print("  Including tudatpy binaries...")
            cmd.extend(tudat_binaries)
        
        # ── Hidden imports ────────────────────────────────────────────────
        # For packages with many C extensions, --collect-all below is the
        # definitive fix. Hidden imports here cover dynamically loaded modules
        # that collect-all alone may miss.
        hidden_imports = [
            # PyQt5 — collect-all not needed, hook handles it
            'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
            'PyQt5.QtPrintSupport', 'PyQt5.QtNetwork', 'PyQt5.QtSvg',
            'PyQt5.QtWebEngineWidgets', 'PyQt5.QtWebEngineCore', 'PyQt5.QtWebChannel',
            # numpy internals — belt-and-suspenders alongside collect-all=numpy
            'numpy', 'numpy.core', 'numpy.lib', 'numpy.fft', 'numpy.linalg',
            'numpy.core._multiarray_tests', 'numpy.core._multiarray_umath',
            # pandas
            'pandas', 'pandas.plotting', 'pandas._libs.tslibs.timedeltas',
            # matplotlib
            'matplotlib', 'matplotlib.pyplot',
            'matplotlib.backends.backend_qt5agg',
            'matplotlib.backends.backend_agg',
            # scipy C extensions commonly missed
            'scipy', 'scipy.interpolate', 'scipy.integrate',
            'scipy.spatial.transform', 'scipy.spatial.transform._rotation_groups',
            'scipy.special._ufuncs_cxx',
            'scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack',
            # astropy
            'astropy', 'astropy.coordinates', 'astropy.time', 'astropy.units',
            'astropy.wcs', 'astropy.io.fits', 'astropy.table',
            # skyfield / sgp4
            'skyfield', 'skyfield.api', 'skyfield.almanac',
            'sgp4', 'sgp4.api',
            # plotly
            'plotly', 'plotly.graph_objs', 'plotly.express',
            # cartopy
            'cartopy', 'cartopy.crs', 'cartopy.feature',
            # utilities
            'tqdm', 'termcolor', 'requests', 'pytz',
            # Pillow
            'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont',
            # pkg_resources — used by pyi_rth_pkgres runtime hook
            'pkg_resources', 'pkg_resources.extern',
        ]

        # Only add tudat hidden imports if it is actually installed
        if tudat_path:
            hidden_imports += [
                'tudatpy', 'tudatpy.kernel', 'tudatpy.kernel.interface',
                'tudatpy.kernel.numerical_simulation',
                'tudatpy.kernel.astro', 'tudatpy.util',
            ]

        for imp in hidden_imports:
            cmd.append(f'--hidden-import={imp}')

        # ── Collect-all ───────────────────────────────────────────────────
        # Use collect-all for packages that have:
        #   (a) data files PyInstaller's static analysis misses, OR
        #   (b) many C extension submodules (numpy, scipy, pandas)
        # This is the DEFINITIVE fix for "ModuleNotFoundError" at runtime
        # for these packages — it bundles everything, not just what's detected.
        collect_all_packages = [
            'PIL',           # .so binaries — prevents decompression errors
            'numpy',         # ALL C extensions: _multiarray_tests, _umath, etc.
            'scipy',         # ALL C extensions: linalg, special, integrate, etc.
            'pandas',        # Extension modules and data files
            'setuptools',    # jaraco/text/Lorem ipsum.txt — crashes without this
            'pkg_resources', # Runtime hook pyi_rth_pkgres dependency
            'astropy',       # FITS data, coordinate frames, unit definitions
            'skyfield',      # Almanac and star catalog data files
            'cartopy',       # Map shapefiles and feature data
            'matplotlib',    # Fonts, style sheets, locale data
        ]
        if tudat_path:
            collect_all_packages.append('tudatpy')

        for pkg in collect_all_packages:
            cmd.append(f'--collect-all={pkg}')

        # ── Excludes ──────────────────────────────────────────────────────
        # ONLY exclude things 100% confirmed not needed — wrong exclusions
        # cause cascading ModuleNotFoundError crashes at runtime.
        excludes = [
            # tkinter / turtle — not used in a PyQt5 app
            'tkinter', 'turtle',
            # Alternative Qt bindings — not used
            'PyQt6', 'PySide2', 'PySide6',
            # QtWebEngine — install via: pip install PyQtWebEngine
            # Removed from excludes since astraa.py imports it directly.
            # If build fails with missing locales/resources, run: pip install PyQtWebEngine
            # Optional compiled speedup for charset_normalizer — pure-Python fallback exists
            'charset_normalizer.md__mypyc',
        ]
        for exc in excludes:
            cmd.append(f'--exclude-module={exc}')

        if self.is_macos:
            cmd.append('--osx-bundle-identifier=com.mynaric.astraa')

        # ── Compression / stripping ───────────────────────────────────────
        # --noupx: prevents UPX from compressing .so files (corrupts PIL binaries)
        # --strip: safe on macOS/Windows; on Linux it corrupts .so shared libs
        if self.is_linux:
            cmd.extend(['--noupx', '--log-level=WARN'])
        else:
            cmd.extend(['--strip', '--noupx', '--log-level=WARN'])

        # Purge UPX from PATH so PyInstaller cannot find it even if installed
        # Use os.pathsep so this works correctly on Windows (;) and Linux/Mac (:)
        env = os.environ.copy()
        path_sep = os.pathsep
        env['PATH'] = path_sep.join(
            p for p in env.get('PATH', '').split(path_sep)
            if 'upx' not in p.lower()
        )

        # ── Fix missing QtWebEngine files in conda+pip mixed environments ──
        # PyQtWebEngine (pip) installs WebEngine files separately from PyQt5 (conda).
        # PyInstaller's hook looks in PyQt5/Qt5/libexec/ and Qt5/translations/ but
        # pip puts them elsewhere. Find and copy them to where the hook expects.
        self._fix_qtwebengine_paths()

        cmd.append(str(main_file))

        print("\n" + "="*70)
        print("⏳ Building executable... (typically 2-5 minutes)")
        print("="*70)
        print()
        
        # Start progress spinner in background thread
        self.building = True
        spinner_thread = threading.Thread(target=self.show_progress_spinner, daemon=True)
        spinner_thread.start()
        
        # Run PyInstaller
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        # Stop spinner
        self.building = False
        spinner_thread.join(timeout=1)
        
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        
        print(f"\n⏱️  Build took: {mins}m {secs}s")
        
        if result.returncode == 0:
            print("✅ Binary built successfully!")
            
            # Show what was created
            if Path('dist/ASTRAA').exists():
                size = Path('dist/ASTRAA').stat().st_size / (1024*1024)
                print(f"📦 Binary size: {size:.1f} MB")
            elif Path('dist/ASTRAA.exe').exists():
                size = Path('dist/ASTRAA.exe').stat().st_size / (1024*1024)
                print(f"📦 Binary size: {size:.1f} MB")
            
            return True
        else:
            print("❌ Build failed!")
            print("\n" + "="*70)
            print("ERROR OUTPUT:")
            print("="*70)
            
            # Parse and show relevant errors
            errors = result.stderr.split('\n')
            
            # Show last 30 lines of error (most relevant)
            print('\n'.join(errors[-30:]))
            
            # Check for common issues
            if 'tudatpy' in result.stderr.lower():
                print("\n⚠️  TUDAT ERROR DETECTED!")
                print("Solutions:")
                print("  1. Make tudat optional in your code")
                print("  2. Install: conda install -c tudat-team tudatpy")
            
            if 'memory' in result.stderr.lower():
                print("\n⚠️  MEMORY ERROR DETECTED!")
                print("Try: Close other applications and retry")
            
            if 'permission' in result.stderr.lower():
                print("\n⚠️  PERMISSION ERROR DETECTED!")
                print("Try: Run with sudo (Linux/Mac) or as Administrator (Windows)")
            
            return False
    
    def _fix_qtwebengine_paths(self):
        """
        PyQtWebEngine (pip) and PyQt5 (conda) install to different locations.
        PyInstaller's hook hard-codes paths inside PyQt5/Qt5/ and crashes when
        QtWebEngineProcess, resources/, and locales/ are missing there.
        This method finds the real files and copies/links them into place.
        """
        import PyQt5
        qt5_dir = Path(PyQt5.__file__).parent / 'Qt5'

        # Dirs that must exist (hook checks them even if empty)
        for d in [
            qt5_dir / 'translations' / 'qtwebengine_locales',
            qt5_dir / 'resources',
            qt5_dir / 'libexec',
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Find QtWebEngineProcess — search common locations
        search_roots = [
            Path(PyQt5.__file__).parent,          # PyQt5 package dir
            Path(sys.executable).parent,           # conda env bin/
            Path(sys.executable).parent.parent,    # conda env root
        ]
        candidate_names = ['QtWebEngineProcess', 'QtWebEngineProcess.exe']
        found = None
        for root in search_roots:
            for name in candidate_names:
                for match in root.rglob(name):
                    found = match
                    break
                if found:
                    break
            if found:
                break

        target = qt5_dir / 'libexec' / 'QtWebEngineProcess'
        if found and found.exists():
            if not target.exists():
                shutil.copy2(str(found), str(target))
                target.chmod(0o755)
                print(f"  ✓ Copied QtWebEngineProcess → {target}")
        else:
            # Create a dummy placeholder so the hook doesn't crash.
            # The real process will be found via PATH at runtime.
            if not target.exists():
                target.write_bytes(b'')
                target.chmod(0o755)
                print(f"  ⚠️  QtWebEngineProcess not found — created placeholder")

        # Copy resources/ and locales/ if found elsewhere
        for src_name, dst_rel in [
            ('qtwebengine_locales', qt5_dir / 'translations' / 'qtwebengine_locales'),
            ('resources',           qt5_dir / 'resources'),
        ]:
            for root in search_roots:
                for match in root.rglob(src_name):
                    if match.is_dir() and match != dst_rel:
                        for f in match.iterdir():
                            dst_f = dst_rel / f.name
                            if not dst_f.exists():
                                shutil.copy2(str(f), str(dst_f))
                        print(f"  ✓ Populated {src_name}/")
                        break

    def find_data_folders(self):
        """Find all data folders to include"""
        data_folders = []
        folders_to_check = [
            'examples', 'attitude_tools', 'basic_tools',
            'pointing_calculations', 'prediction_methods',
            'tudat_tools', 'analyses', 'resources'
        ]
        
        for folder_name in folders_to_check:
            folder_path = self.root / folder_name
            if folder_path.exists() and folder_path.is_dir():
                sep = ';' if self.is_windows else ':'
                data_folders.append(f'--add-data={folder_name}{sep}{folder_name}')
                print(f"  Found: {folder_name}/")
        
        return data_folders
    
    def create_distribution(self):
        """Package the binary"""
        print("\n📦 Creating distribution package...")
        
        if self.is_windows:
            binary = Path('dist') / 'ASTRAA.exe'
        elif self.is_macos:
            binary = Path('dist') / 'ASTRAA.app'
            if not binary.exists():
                binary = Path('dist') / 'ASTRAA'
        else:
            binary = Path('dist') / 'ASTRAA'
        
        if not binary.exists():
            print(f"❌ Binary not found: {binary}")
            return False
        
        self.dist_dir.mkdir(exist_ok=True)
        
        platform_name = {
            'Darwin': 'macOS',
            'Linux': 'linux', 
            'Windows': 'windows'
        }.get(self.system, 'unknown')
        
        release_name = f'ASTRAA-v{VERSION}-{platform_name}-{self.build_date}'
        release_folder = self.dist_dir / release_name
        release_folder.mkdir(exist_ok=True)
        
        print(f"  Copying binary to release folder...")
        
        if binary.suffix == '.app':
            shutil.copytree(binary, release_folder / binary.name)
        else:
            shutil.copy2(binary, release_folder)
        
        print(f"✓ Binary: {binary.name}")
        
        self._create_readme(release_folder, platform_name)
        
        print(f"  Creating archive...")
        archive_path = self._create_archive(release_folder)
        
        return archive_path
    
    def _create_readme(self, folder, platform_name):
        """Create README"""
        readme = folder / 'README.txt'
        readme.write_text(f'''ASTRAA v{VERSION} - {platform_name}
Build Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

NO INSTALLATION REQUIRED!
========================
This is a standalone executable with everything bundled:
- Python 3.10 interpreter (built-in)
- All required libraries (built-in)
- Your code (compiled and protected)

Users DO NOT need:
- Python installation
- pip or conda
- Any packages
- Virtual environments

Just run the executable!

USAGE
=====
{platform_name}: Double-click ASTRAA or run ./ASTRAA

WHAT'S INCLUDED
===============
✓ Python 3.10 interpreter
✓ NumPy, Pandas, Matplotlib, SciPy
✓ PyQt5, Plotly
✓ Astropy, Skyfield, SGP4
✓ Tudatpy (if available during build)
✓ All your custom modules

SYSTEM REQUIREMENTS
===================
{platform_name} only - no Python needed!
RAM: 4GB minimum
Disk: 500MB free space

COPYRIGHT © 2025 Mynaric
''')
        print(f"✓ README: README.txt")
    
    def _create_archive(self, folder):
        """Create archive"""
        base_name = folder.stem
        fmt = 'gztar' if not self.is_windows else 'zip'
        archive = shutil.make_archive(
            str(self.dist_dir / base_name),
            fmt,
            folder.parent,
            folder.name
        )
        
        # Show archive size
        archive_path = Path(archive)
        size = archive_path.stat().st_size / (1024*1024)
        print(f"✓ Archive: {archive_path.name} ({size:.1f} MB)")
        
        return archive_path
    
    def clean(self):
        """Clean build artifacts"""
        print("🧹 Cleaning old builds...")
        for d in ['build', 'dist', '__pycache__']:
            path = Path(d)
            if path.exists():
                shutil.rmtree(path)
        for spec in Path('.').glob('*.spec'):
            spec.unlink()
        print("✓ Cleaned")
    
    def find_main_file(self):
        """Find astraa.py"""
        main = self.root / MAIN_FILE
        if main.exists():
            return main
        for path in self.root.rglob(MAIN_FILE):
            if '__pycache__' not in str(path):
                return path
        print(f"❌ Could not find {MAIN_FILE}")
        sys.exit(1)
    
    def check_dependencies(self):
        """Install PyInstaller if needed"""
        try:
            import PyInstaller
            print("✓ PyInstaller found")
        except ImportError:
            print("📦 Installing PyInstaller...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
            print("✓ PyInstaller installed")
    
    def run(self):
        """Run build process"""
        print("="*70)
        print(f"  ASTRAA Builder with Progress Indicator")
        print(f"  Platform: {self.system}")
        print(f"  Version: {VERSION}")
        print("="*70)
        print()
        
        self.check_dependencies()
        print()
        
        main_file = self.find_main_file()
        print(f"✓ Main file: {main_file}")
        print()
        
        self.clean()
        print()
        
        if not self.build_binary(main_file):
            print("\n❌ BUILD FAILED - See errors above")
            return False
        
        archive = self.create_distribution()
        if not archive:
            print("\n❌ PACKAGING FAILED")
            return False
        
        print()
        print("="*70)
        print("  ✅ BUILD COMPLETE!")
        print("="*70)
        print(f"\n📦 Package: {archive}")
        print(f"📁 Location: {archive.parent}")
        print()
        print("✅ Users DON'T need Python, conda, or ANY packages!")
        print("✅ Everything is bundled in the executable!")
        print()
        print("Next steps:")
        print("  1. Test the binary")
        print(f"  2. Upload {archive.name} to your server")
        print("  3. Share download link with users")
        print()
        
        return True

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              ASTRAA Binary Builder                           ║
║              With Real-Time Progress                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    builder = ASTRAABuilder()
    success = builder.run()
    sys.exit(0 if success else 1)