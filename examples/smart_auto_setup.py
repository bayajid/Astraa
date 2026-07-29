#!/usr/bin/env python3
"""
ASTRAA Smart Auto-Setup
This script ACTUALLY finds your files automatically and organizes everything
No manual work required - just run it!
"""

import os
import shutil
import sys
from pathlib import Path
import re

class SmartAstraaSetup:
    def __init__(self):
        self.source_root = Path.cwd()
        self.package_root = self.source_root / 'astraa_package'
        self.astraa_dir = self.package_root / 'astraa'
        self.file_cache = {}
        
    def find_file(self, filename):
        """Intelligently search for a file"""
        if filename in self.file_cache:
            return self.file_cache[filename]
        
        # Search strategy:
        # 1. Current directory
        # 2. Parent directories (up to 3 levels)
        # 3. All subdirectories (excluding common excludes)
        
        search_paths = [
            self.source_root,
            self.source_root.parent,
            self.source_root.parent.parent,
        ]
        
        for search_path in search_paths:
            for root, dirs, files in os.walk(search_path):
                # Skip these directories
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', 'env', 
                                                          'build', 'dist', 'astraa_package', 
                                                          '.idea', '.vscode']]
                
                if filename in files:
                    found_path = Path(root) / filename
                    self.file_cache[filename] = found_path
                    return found_path
        
        return None
    
    def find_module_by_import(self, import_line):
        """Extract module info from import statement"""
        # Handle different import styles:
        # import module.submodule as alias
        # from module import submodule
        # from module.submodule import something
        
        patterns = [
            r'from\s+([\w\.]+)\s+import',
            r'import\s+([\w\.]+)\s+as',
            r'import\s+([\w\.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, import_line)
            if match:
                module_path = match.group(1)
                # Convert to file path
                parts = module_path.split('.')
                filename = parts[-1] + '.py'
                return filename, module_path
        
        return None, None
    
    def analyze_imports(self):
        """Analyze astraa.py to find all required modules"""
        print("🔍 Analyzing your code to find required modules...")
        
        astraa_file = self.find_file('astraa.py')
        if not astraa_file:
            print("❌ Cannot find astraa.py!")
            sys.exit(1)
        
        print(f"✓ Found astraa.py at: {astraa_file}")
        
        with open(astraa_file, 'r') as f:
            content = f.read()
        
        # Find all import statements
        import_lines = re.findall(r'^(?:from|import)\s+.*$', content, re.MULTILINE)
        
        required_modules = {}
        
        for line in import_lines:
            # Skip standard library and external packages
            if any(skip in line for skip in ['PyQt5', 'numpy', 'pandas', 'matplotlib', 
                                               'scipy', 'plotly', 'tudatpy', 'astropy',
                                               'skyfield', 'sgp4', 'cartopy', 'sys', 
                                               'os', 'pathlib', 'datetime']):
                continue
            
            filename, module_path = self.find_module_by_import(line)
            if filename and module_path:
                required_modules[filename] = {
                    'module_path': module_path,
                    'import_line': line.strip()
                }
        
        print(f"✓ Found {len(required_modules)} custom modules to locate")
        return required_modules
    
    def create_structure(self):
        """Create the package structure"""
        print("\n📁 Creating package structure...")
        
        folders = [
            self.astraa_dir / 'lib' / 'attitude_tools',
            self.astraa_dir / 'lib' / 'basic_tools',
            self.astraa_dir / 'lib' / 'pointing_calculations',
            self.astraa_dir / 'lib' / 'prediction_methods',
            self.astraa_dir / 'lib' / 'tudat_tools' / 'data_processing',
            self.astraa_dir / 'lib' / 'tudat_tools' / 'astro_simulations',
            self.astraa_dir / 'lib' / 'analyses' / 'attitude_predictions',
            self.astraa_dir / 'resources' / 'images',
            self.astraa_dir / 'resources' / 'sat',
            self.astraa_dir / 'resources' / 'templates',
            self.astraa_dir / 'resources' / 'config',
            self.astraa_dir / 'data' / 'input_data',
            self.astraa_dir / 'data' / 'output_data',
            self.package_root / 'tests',
        ]
        
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / '__init__.py').touch()
        
        print("✓ Structure created")
    
    def smart_copy_files(self, required_modules):
        """Intelligently find and copy all required files"""
        print("\n📋 Finding and copying your files...")
        
        # Mapping of module names to target directories
        target_map = {
            'rotations': 'lib/attitude_tools',
            'conversions': 'lib/attitude_tools',
            'time_conversion': 'lib/basic_tools',
            'in_out': 'lib/basic_tools',
            'ae_calculation': 'lib/pointing_calculations',
            'interpolators': 'lib/prediction_methods',
            'j2propagator': 'lib/prediction_methods',
            'simulation_utilities': 'lib/tudat_tools',
            'tudat_converter': 'lib/tudat_tools',
            'data_processing_utilities': 'lib/tudat_tools/data_processing',
            'data_loading': 'lib/tudat_tools/data_processing',
            'astro_moon_rooftop_azel': 'lib/tudat_tools/astro_simulations',
            'attitude_prediction_utlities': 'lib/analyses/attitude_predictions',
            'tle_to_j2000': 'lib',
            'quaternion_slerp_squad': 'lib',
        }
        
        copied = 0
        not_found = []
        
        for filename in required_modules.keys():
            module_name = filename.replace('.py', '')
            
            # Find the file
            found_path = self.find_file(filename)
            
            if found_path:
                # Determine target directory
                target_subdir = target_map.get(module_name, 'lib')
                target_dir = self.astraa_dir / target_subdir
                target_path = target_dir / filename
                
                # Copy file
                shutil.copy2(found_path, target_path)
                print(f"  ✓ {filename}")
                print(f"    From: {found_path}")
                print(f"    To:   {target_path.relative_to(self.package_root)}")
                copied += 1
            else:
                not_found.append(filename)
                print(f"  ⚠ {filename} - NOT FOUND")
        
        # Copy main file
        main_file = self.find_file('astraa.py')
        if main_file:
            shutil.copy2(main_file, self.astraa_dir / 'astraa_gui.py')
            print(f"  ✓ astraa.py → astraa_gui.py")
            copied += 1
        
        # Copy resources
        self.copy_resources()
        
        print(f"\n✓ Copied {copied} files")
        if not_found:
            print(f"⚠ Could not find {len(not_found)} files: {', '.join(not_found)}")
        
        return copied > 0
    
    def copy_resources(self):
        """Find and copy resource files"""
        print("\n📦 Finding resource files...")
        
        resource_files = {
            'astraa_small.png': 'resources/images',
            'ICESat-2.glb': 'resources/sat',
            'cesium_3D_trajectory_copy.html': 'resources/templates',
            'satellite_config.json': 'resources/config',
            'ground_stations.json': 'resources/config',
            'attitude_settings.json': 'resources/config',
            'simulation_parameters.json': 'resources/config',
        }
        
        for filename, target_subdir in resource_files.items():
            found_path = self.find_file(filename)
            if found_path:
                target_dir = self.astraa_dir / target_subdir
                target_path = target_dir / filename
                shutil.copy2(found_path, target_path)
                print(f"  ✓ {filename}")
    
    def create_config_files(self):
        """Create setup.py, requirements.txt, etc."""
        print("\n📄 Creating configuration files...")
        
        # setup.py
        (self.package_root / 'setup.py').write_text('''from setuptools import setup, find_packages

setup(
    name="astraa",
    version="1.0.0",
    author="Dr. Bayajid Khan",
    author_email="your.email@mynaric.com",
    description="ASTRAA - Satellite Trajectory and Attitude Analysis",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "PyQt5>=5.15.0",
        "PyQtWebEngine>=5.15.0",
        "plotly>=5.14.0",
        "tqdm>=4.65.0",
        "termcolor>=2.3.0",
        "sgp4>=2.21",
        "astropy>=5.2.0",
        "skyfield>=1.45",
        "pytz>=2023.3",
        "requests>=2.31.0",
        "cartopy>=0.21.0",
    ],
    entry_points={
        'console_scripts': ['astraa-gui=astraa.astraa_gui:main'],
        'gui_scripts': ['astraa=astraa.astraa_gui:main'],
    },
    package_data={'astraa': ['resources/**/*']},
    include_package_data=True,
)
''')
        
        # requirements.txt
        (self.package_root / 'requirements.txt').write_text('''numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
PyQt5>=5.15.0
PyQtWebEngine>=5.15.0
plotly>=5.14.0
tqdm>=4.65.0
termcolor>=2.3.0
sgp4>=2.21
astropy>=5.2.0
skyfield>=1.45
pytz>=2023.3
requests>=2.31.0
cartopy>=0.21.0
''')
        
        # .gitignore
        (self.package_root / '.gitignore').write_text('''__pycache__/
*.py[cod]
build/
dist/
*.egg-info/
*.tar.gz
*.zip
*.exe
venv/
''')
        
        # MANIFEST.in
        (self.package_root / 'MANIFEST.in').write_text('''include README.md
include requirements.txt
recursive-include astraa/resources *
''')
        
        # README.md
        (self.package_root / 'README.md').write_text('''# ASTRAA

Satellite Trajectory and Attitude Analysis

## Installation
```bash
pip install -e .
```

## Usage
```bash
astraa-gui
```
''')
        
        print("✓ Configuration files created")
    
    def update_imports_info(self):
        """Create a file explaining what needs to be updated"""
        info = """
ASTRAA Package - Import Updates Needed
======================================

Your package structure has been created automatically!

IMPORTANT: You need to update the imports in astraa_gui.py

Find and replace in astraa/astraa_gui.py:
-----------------------------------------

OLD IMPORTS (remove):
  import attitude_tools.rotations as rot
  import basic_tools.time_conversion as t_conv
  etc.

NEW IMPORTS (use):
  from .lib.attitude_tools import rotations as rot
  from .lib.basic_tools import time_conversion as t_conv
  etc.

Quick Fix Command:
------------------
cd astraa_package
sed -i 's/import attitude_tools\./from .lib.attitude_tools import /g' astraa/astraa_gui.py
sed -i 's/import basic_tools\./from .lib.basic_tools import /g' astraa/astraa_gui.py
sed -i 's/import pointing_calculations\./from .lib.pointing_calculations import /g' astraa/astraa_gui.py
sed -i 's/import prediction_methods\./from .lib.prediction_methods import /g' astraa/astraa_gui.py
sed -i 's/import tudat_tools\./from .lib.tudat_tools import /g' astraa/astraa_gui.py
sed -i 's/import analyses\./from .lib.analyses import /g' astraa/astraa_gui.py

After updating imports:
-----------------------
pip install -e .
astraa-gui
"""
        
        (self.package_root / 'UPDATE_IMPORTS.txt').write_text(info)
    
    def run(self):
        """Run the complete setup"""
        print("="*60)
        print("  ASTRAA Smart Auto-Setup")
        print("  Automatically finds and organizes your files")
        print("="*60)
        
        # Analyze imports to understand structure
        required_modules = self.analyze_imports()
        
        # Create structure
        self.create_structure()
        
        # Find and copy files
        success = self.smart_copy_files(required_modules)
        
        if not success:
            print("\n❌ No files were copied!")
            print("Please ensure you're running this from the correct directory")
            sys.exit(1)
        
        # Create config files
        self.create_config_files()
        
        # Create import update instructions
        self.update_imports_info()
        
        print("\n" + "="*60)
        print("  ✓ SETUP COMPLETE!")
        print("="*60)
        print(f"\nPackage created at: {self.package_root}")
        print("\nNext steps:")
        print("  1. cd astraa_package")
        print("  2. Read UPDATE_IMPORTS.txt")
        print("  3. Update imports in astraa/astraa_gui.py")
        print("  4. pip install -e .")
        print("  5. astraa-gui")
        print("\n" + "="*60)


if __name__ == '__main__':
    setup = SmartAstraaSetup()
    setup.run()
