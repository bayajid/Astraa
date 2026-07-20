from setuptools import setup, find_packages

setup(
    name='astraa_pkg',
    version='0.1.0',
    description='Standalone astraa package with all dependencies',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'PyQt5',
        'PyQtWebEngine',
        'plotly',
        'scipy',
        'tudatpy',
        'cartopy',
        'skyfield',
        'astropy',
        'pytz',
        'requests'
    ],
    include_package_data=True,
    python_requires='>=3.7',
)
