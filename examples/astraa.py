from pathlib import Path
import os
import sys

prefix = sys.prefix

os.environ["QT_PLUGIN_PATH"] = os.path.join(prefix, "plugins")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(prefix, "plugins", "platforms")

os.environ["QTWEBENGINEPROCESS_PATH"] = os.path.join(prefix, "libexec", "QtWebEngineProcess")

os.environ["QTWEBENGINE_RESOURCES_PATH"] = os.path.join(prefix, "resources")
os.environ["QTWEBENGINE_LOCALES_PATH"] = os.path.join(prefix, "translations")


_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
for _path in (_REPO_ROOT, _EXAMPLES_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

#from turtle import position, width
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QCheckBox,QFileDialog,
                            QHBoxLayout, QPushButton, QLabel, QComboBox, QLineEdit, QDateTimeEdit,
                            QSpinBox, QDoubleSpinBox, QTabWidget, QGroupBox, QFormLayout, QDateEdit, QSplashScreen, QAction, QMessageBox, QFrame, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, QUrl, QSize, QDate,QDateTime
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings, QWebEnginePage
# from PyQt5.QtWebEngine import QtWebEngine

import tempfile, os, webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from termcolor import colored
from sgp4.api import Satrec, jday

import json
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
# Load tudatpy modules
# from tudatpy.kernel.interface import spice
# # from tudatpy.data import save2txt
# from tudatpy.kernel import dynamics
# from tudatpy.kernel.dynamics import environment_setup
# from tudatpy.kernel.dynamics import propagation_setup
# from tudatpy.kernel.astro import element_conversion
# # from tudatpy.kernel import constants
# from tudatpy.util import result2array

from tudatpy.kernel import dynamics
from tudatpy.kernel.dynamics import environment_setup
from tudatpy.kernel.dynamics import propagation_setup



import tudat_tools.simulation_utilities as util
import tudat_tools.data_processing.data_processing_utilities as dputil
import tudat_tools.tudat_converter as tudatconv
import tudat_tools.data_processing.data_loading as load
from tudat_tools.astro_simulations.astro_moon_rooftop_azel import ae_roof2sun


# Import custom modules
# import basic_tools.vector_operations as vec_calc
# import astronomy_tools.constants as const
# import astronomy_tools.astro_targets as where_sun
# import attitude_tools.attitude_simulation as att_sim
import attitude_tools.rotations as rot
# import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as io

# import basic_tools.in_out as out
import pointing_calculations.ae_calculation as ae_calc
import prediction_methods.interpolators as interp
import prediction_methods.j2propagator as j2prop

import attitude_tools.conversions as att_conv
import analyses.attitude_predictions.attitude_prediction_utlities as att_pred

from cartopy import crs as ccrs
from cartopy.feature import LAND, COASTLINE, BORDERS
import json, importlib
import csv
from time import perf_counter
from tle_to_j2000 import propagate_and_rotate_tle
import quaternion_slerp_squad as quat_slerp

# New MEKF approach imports
from MEKF import MEKFComparator
import time


# New imports for Moon Phase calculation
from skyfield import  api as skyapi
from datetime import timedelta, datetime, timezone
import datetime as dt
from glob import glob
from zoneinfo import ZoneInfo

# from skyfield.framelib import ecliptic_frame
from skyfield.trigonometry import position_angle_of

# astropy stuff
from astropy.time import Time, TimeDelta
from astropy.coordinates import TEME,EarthLocation, AltAz, CartesianRepresentation,CartesianDifferential, GCRS, SkyCoord, ITRS, get_sun, get_moon
from astropy import units as u
from scipy.spatial.transform import Rotation as R


CET = ZoneInfo("Europe/Berlin") # timezone('Europe/Berlin')

_MIN_PYTHON = (3, 10)
_MAX_PYTHON = (3, 13)
current_version = sys.version_info[:3]

if current_version < _MIN_PYTHON or current_version >= _MAX_PYTHON:
    print(
        colored(
            f"Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ (below {_MAX_PYTHON[0]}.{_MAX_PYTHON[1]}) is required; "
            f"you are using {current_version[0]}.{current_version[1]}.{current_version[2]}.",
            "yellow",
        )
    )
else:
    print(colored("Python version check passed.", "green"))
class DebugWebEnginePage(QWebEnginePage):
    """Custom QWebEnginePage to capture JavaScript console messages"""
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS Console: {message} (Line: {lineNumber}, Source: {sourceID})")
class AstraaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASTRAA- Astropynaric Satellite Trajectory and Attitude Analysis")
        self.setGeometry(100, 100, 1600, 900)

        ## Folder management
        ## Check if folders exist, if not create them
        _EXAMPLES_DIR.joinpath("input_data").mkdir(parents=True, exist_ok=True)
        _EXAMPLES_DIR.joinpath("output_data").mkdir(parents=True, exist_ok=True)

        ## Set folder paths (anchored to this script, not the process cwd)
        self.datadir = str(_EXAMPLES_DIR / "input_data")
        self.outputdir = str(_EXAMPLES_DIR / "output_data")
        
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
            print(f"Folder '{self.outputdir}' created.")

        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
            print(f"Folder '{self.datadir}' created.")
        
        # Initialize main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create split panel layout first
        split_layout = QHBoxLayout()
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(0)
        
        # Left panel (now including logo)
        left_panel = QWidget()
        left_panel.setMaximumWidth(500)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # Add logo section to left panel
        logo_layout = QHBoxLayout()
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(0)
        
        # ASTRAA logo
        astraa_label = QLabel()
        astraa_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        astraa_path = os.path.join(os.path.dirname(__file__), 'astraa_small.png')
        if os.path.exists(astraa_path):
            pixmap = QPixmap(astraa_path)
            new_width = int(pixmap.width() * 1)
            new_height = int(pixmap.height() * 1)
            pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            astraa_label.setPixmap(pixmap)
        logo_layout.addWidget(astraa_label)
        logo_layout.addStretch()
        left_layout.addLayout(logo_layout)
        
        # Add title label below the logo in left panel
        title_label = QLabel("ASTRAA\nAstroPynaric Satellite Trajectory and Attitude Analysis")
        title_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 5px;")
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        title_label.setFixedWidth(left_panel.width())
        left_layout.addWidget(title_label)
        
        # Add tab widget
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)
        
        self.orbit_tab = QWidget()
        self.attitude_tab = QWidget()
        self.link_tab = QWidget()
        self.ogs_tab = QWidget()
        self.moon_phase_tab = QWidget() # New tab widget
        
        self.tabs.addTab(self.orbit_tab, "Orbit Simulation")
        self.tabs.addTab(self.attitude_tab, "Attitude Generation")
        self.tabs.addTab(self.link_tab, "Link Analysis")
        self.tabs.addTab(self.ogs_tab, "OGS")
        self.tabs.addTab(self.moon_phase_tab, "Sun/Moon") # Add new tab
        
        split_layout.addWidget(left_panel, 30)
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        self.graphics_tabs = QTabWidget()
        right_layout.addWidget(self.graphics_tabs)
        
        self.orbit_graphics = QWidget()
        self.attitude_graphics = QWidget()
        self.link_graphics = QWidget()
        self.pe_graphics = QWidget()
        self.moon_graphics = QWidget() # New graphics tab widget
        self.ogs_graphics = QWidget() # New graphics tab for OGS
        
        self.graphics_tabs.addTab(self.orbit_graphics, "Orbit Visualization")
        self.graphics_tabs.addTab(self.attitude_graphics, "Attitude Visualization")
        self.graphics_tabs.addTab(self.link_graphics, "Link Analysis Visualization")
        self.graphics_tabs.addTab(self.ogs_graphics, "Ground Station Visibility") # Add new graphics tab
        self.graphics_tabs.addTab(self.moon_graphics, "Sun/Moon Phase") # Add new graphics tab
        self.graphics_tabs.addTab(self.pe_graphics, "Pointing Error Visualization")
        
        
        
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        split_layout.addWidget(right_panel, 70)
        
        main_layout.addLayout(split_layout)
        
        self.setup_orbit_tab()
        self.setup_attitude_tab()
        self.setup_link_tab()
        self.setup_ogs_tab()
        self.setup_moon_tab() # Call setup for new tab
        
        self.simulation_data = {}
        self.attitude_data = {}
        self.link_data = {}
        self.pe_data = {}
        
        # Initialize Tudat Predictor for coordinate transformations
        self.tudat_converter = tudatconv.tudat_predictor()
        
        # Add a menu bar
        menubar = self.menuBar()
        info_menu = menubar.addMenu("&Info") # & creates a shortcut (Alt+I)

        # Add actions to the Info menu
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        info_menu.addAction(about_action)
        
    def on_tab_changed(self, index):
        self.graphics_tabs.setCurrentIndex(index)
        
    def show_about_dialog(self):
        """Display an about dialog for the application."""
        QMessageBox.about(self, "About ASTRAA",
            rf"""
            <img src="{os.path.join(os.path.dirname(__file__), 'astraa_small.png')}" width="100"><br>
            <b>ASTRAA (Astropynaric Satellite Trajectory and Attitude Analysis)</b><br>
            Version 1.0<br>
            <br>
            This application provides tools for:<br>
            <ul>
                <li>Orbit Simulation</li>
                <li>Attitude Generation</li>
                <li>Link Analysis</li>
                <li>Moon Phase Calculation</li>
            </ul><br>
            Powered by: Tudat space<br>
             Developed by: Dr. Bayajid Khan<br>
            &copy; Mynaric, 2025
            """)
        
    def setup_orbit_tab(self):
        """Setup the orbit simulation tab"""
        layout = QVBoxLayout(self.orbit_tab)
        
        # 1. Simulation controls block (group box)
        control_group = QGroupBox("Simulation Controls")
        control_layout = QFormLayout(control_group)
        
        # Add simulation parameters in rows
        self.sim_time = QSpinBox()
        self.sim_time.setRange(1,72)
        self.sim_time.setValue(1)
        self.sim_time.setSuffix(" hours")
        control_layout.addRow("Simulation Time:", self.sim_time)
        
        self.precision = QComboBox()
        self.precision.addItems(["Medium Precision","High Precision",  "J2 Only"])
        control_layout.addRow("Precision:", self.precision)
        
        self.sat_count = QSpinBox()
        self.sat_count.setRange(1, 10)
        self.sat_count.setValue(6)
        control_layout.addRow("Number of Satellites:", self.sat_count)
        
        # Add a dropdown for simulation type (if not already present)
        # self.sim_type_dropdown = QComboBox()
        # self.sim_type_dropdown.addItems(["High Precision", "Medium Precision", "J2 Only"])
        # control_layout.addRow("Simulation Type:", self.sim_type_dropdown)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 2. Run Orbit Simulation button
        self.run_sim_button = QPushButton("Run Orbit Simulation")
        self.run_sim_button.clicked.connect(self.run_orbit_simulation)
        layout.addWidget(self.run_sim_button)
        
        # 3. Simulation info text area at the bottom
        self.sim_info_text = QTextEdit()
        self.sim_info_text.setReadOnly(True)
        layout.addWidget(self.sim_info_text)
        
        # Setup graphics tab
        graphics_layout = QVBoxLayout(self.orbit_graphics)
        self.orbit_plot = QWebEngineView()
        # Set custom page for console logging
        #self.orbit_plot.setPage(DebugWebEnginePage(self.orbit_plot))
        # Enable WebGL and related settings
        # QtWebEngine.initialize()  # Ensure WebEngine is initialized
        settings = self.orbit_plot.settings()
        settings.setAttribute(QWebEngineSettings.WebGLEnabled, True) 
        settings.setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, False)  # Disable GPU canvas
        settings.setAttribute(QWebEngineSettings.PluginsEnabled, True)
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.ShowScrollBars, False)
        graphics_layout.addWidget(self.orbit_plot)
        
        # Load simulation parameters JSON
        self.sim_params_json = {}
        try:
            with open(os.path.join("examples", "input_data", "simulation_parameters.json"), "r") as f:
                self.sim_params_json = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not load simulation_parameters.json: {e}")

        # Map precision dropdown values to JSON keys
        self.precision_map = {
            "High Precision": "High Precision",
            "Medium Precision": "Medium Precision",
            "J2 Only": "J2 Only"
        }
        # Connect precision dropdown change to update info text
        self.precision.currentTextChanged.connect(self.update_sim_info_from_precision)
        # Set initial info text
        self.update_sim_info_from_precision(self.precision.currentText())
        
    def update_sim_info_from_precision(self, precision_text):
        # Map the precision dropdown value to the JSON key
        sim_type = (self.precision_map.get(precision_text))
        params = self.sim_params_json.get(sim_type, {})
        self.update_sim_info_text(params)

    def update_sim_info_text(self, simulation_parameters):
        # Display the required fields, replacing missing values with 0
        def get_val(key):
            v = simulation_parameters.get(key, 0)
            if v is None:
                return 0
            if isinstance(v, list) or isinstance(v, dict):
                return str(v)
            return v
        fields = [
            "Percision",
            "accelerations_setting", 
            "integrator", 
            "time_step", 
            "Max_total_sats", 
            "n_function_evaluations",
            "mass [kg]" ,
            "reference_area [m³]",
            "drag_coefficient" ,
            "SRP_coefficient" 
        ]
        text = "Simulation Parameters:\n"
        for f in fields:
            text += f"{f}: {get_val(f)}\n"
        self.sim_info_text.setPlainText(text)
        
    def setup_attitude_tab(self):
        """Setup the attitude generation tab"""
        layout = QVBoxLayout(self.attitude_tab)
        
        # Control panel
        control_group = QGroupBox("Attitude Controls")
        control_layout = QFormLayout()  # Changed to QFormLayout for rows
        
        # Add settings selection in rows
        self.settings_combo = QComboBox()
        self.settings_combo.addItems([
            "rocketlab_march",
            "rotate_all_axes",
            "rotate_swap",
            "rotate_yaw",
            "rotate_all_pred084",
            "rotate_azel_octhw467",
            "custom",
        ])
        control_layout.addRow("Settings:", self.settings_combo)
        
        self.roll = QDoubleSpinBox()
        self.roll.setRange(-180, 180)
        self.roll.setValue(5)
        self.roll.setSuffix("°")
        control_layout.addRow("Roll:", self.roll)
        
        self.pitch = QDoubleSpinBox()
        self.pitch.setRange(-180, 180)
        self.pitch.setValue(10)
        self.pitch.setSuffix("°")
        control_layout.addRow("Pitch:", self.pitch)
        
        self.yaw = QDoubleSpinBox()
        self.yaw.setRange(-180, 180)
        self.yaw.setValue(45)
        self.yaw.setSuffix("°")
        control_layout.addRow("Yaw:", self.yaw)


        # Create a horizontal layout for the buttons
        checkbox_layout = QHBoxLayout()
        
        # Create Attitude sign swap label
        sign_swap_checkbox = QCheckBox("Sign swap")
        # Connect checkbox signal
        self.sign_swap_flag = False
        sign_swap_checkbox.stateChanged.connect(lambda state: print(f"Sign-swap Flag = {setattr(self, 'sign_swap_flag', state == 2) or self.sign_swap_flag}"))
        checkbox_layout.addWidget(sign_swap_checkbox)

        # Create Jerk label
        jerk_checkbox = QCheckBox("Add jerk motion!")
        # Connect checkbox signal
        self.jerk_flag = False
        jerk_checkbox.stateChanged.connect(lambda state: print(f"Jerk Flag = {setattr(self, 'jerk_flag', state == 2) or self.jerk_flag}"))
        checkbox_layout.addWidget(jerk_checkbox)

        # Create Integrator dropdown
        quaternion_integrator= QComboBox()
        quaternion_integrator.addItems(['RK4', 'Rapid 4th-Order'])#(["SLERP","MOD-CUBIC-SPLINE","CUBIC-SPLINE"])
        integration_label = QLabel("Integrator:")
        checkbox_layout.addWidget(integration_label)
        checkbox_layout.addWidget(quaternion_integrator)
    
        self.active_integrator = quaternion_integrator.currentText()     
        quaternion_integrator.currentTextChanged.connect(self.update_quaternion_integrator)
        
        # quaternion_integrator.currentTextChanged.connect(
        #     lambda text: setattr(self, 'Active_integrator', text)
        # )
        
        #print(f"Active quaternion interpolator: {self.active_integrator}")


        # Connect checkbox signal
        # self.jerk_flag = False
        # jerk_checkbox.stateChanged.connect(lambda state: print(f"Jerk Flag = {setattr(self, 'jerk_flag', state == 2) or self.jerk_flag}"))
        # checkbox_layout.addWidget(jerk_checkbox)

        # Add the button layout to the form
        control_layout.addRow(checkbox_layout)

        generate_button = QPushButton("Generate Attitude")
        generate_button.clicked.connect(self.generate_attitude_selector)
        control_layout.addRow(generate_button)  # Button in its own row
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Add Pointing Error Controls below Attitude Controls
        pe_group = QGroupBox("Pointing Error Controls")
        pe_layout = QFormLayout()

        
        PE_checkbox = QHBoxLayout()
        rsepect_rate_checkbox = QCheckBox("RSEPECT rate!")
        # Connect checkbox signal
        self.rsepect_rate_flag = False
        rsepect_rate_checkbox.stateChanged.connect(lambda state: print(f"RSEPECT Rate Flag = {setattr(self, 'rsepect_rate_flag', state == 2) or self.rsepect_rate_flag}"))
        PE_checkbox.addWidget(rsepect_rate_checkbox)

        # Create Interpolator dropdown
        quaternion_interpolator= QComboBox()
        quaternion_interpolator.addItems(["SLERP","MOD-CUBIC-SPLINE","CUBIC-SPLINE"])
        interpolation_label = QLabel("Interpolator:")
        PE_checkbox.addWidget(interpolation_label)
        PE_checkbox.addWidget(quaternion_interpolator)

        self.active_interpolator = quaternion_interpolator.currentText()     
        quaternion_interpolator.currentTextChanged.connect(self.update_quaternion_interpolator)
        # quaternion_interpolator.currentTextChanged.connect(
        #     lambda text: setattr(self, 'Active_interpolator', text)
        # )

        
        pe_layout.addRow(PE_checkbox)

        self.update_rate = QComboBox()
        for val in [1, 2, 5, 10]:
            self.update_rate.addItem(f"{val} Hz", val)
        pe_layout.addRow("Update Rate:", self.update_rate)

        self.latency = QComboBox()
        for val in [0, 1, 2, 3, 4]:
            self.latency.addItem(f"{val} s", val)
        pe_layout.addRow("Latency:", self.latency)

        calculate_button = QPushButton("Calculate PE")
        if self.active_interpolator == "CUBIC-SPLINE":
            calculate_button.clicked.connect(lambda:self.calculate_pe(self.active_interpolator))
        else:
            calculate_button.clicked.connect(lambda:self.calculate_pe_new(self.active_interpolator))
        pe_layout.addRow(calculate_button)

        # Add Plot button
        plot_button = QPushButton("Plot")
        plot_button.clicked.connect(self.update_attitude_visualization)
        plot_button.clicked.connect(self.plot_pe_from_csv)
        pe_layout.addRow(plot_button)

       
        pe_group.setLayout(pe_layout)
        layout.addWidget(pe_group)
        
        # Setup graphics tab for attitude
        graphics_layout = QVBoxLayout()
        self.attitude_plot = QWebEngineView()
        self.attitude_plot.settings().setAttribute(QWebEngineSettings.WebGLEnabled, True)
        self.attitude_plot.settings().setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, True)
        self.attitude_plot.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.attitude_plot.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        graphics_layout.addWidget(self.attitude_plot)
        self.attitude_graphics.setLayout(graphics_layout)

        # Setup graphics tab for PE
        graphics_layout1 = QVBoxLayout()
        self.pe_plot = QWebEngineView() # Store as instance variable
        self.pe_plot.settings().setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, True)
        self.pe_plot.settings().setAttribute(QWebEngineSettings.WebGLEnabled, True)
        self.pe_plot.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.pe_plot.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        graphics_layout1.addWidget(self.pe_plot)
        self.pe_graphics.setLayout(graphics_layout1)
        
    def setup_link_tab(self):
        """Setup the link analysis tab"""
        layout = QVBoxLayout(self.link_tab)
        
        # Control panel
        control_group = QGroupBox("Link Analysis Controls")
        control_layout = QFormLayout()  # Changed to QFormLayout for rows
        
        # Add satellite selection in rows
        satellite_group = QGroupBox("Satellite Selection")
        satellite_layout = QFormLayout(satellite_group)  # Use QFormLayout for proper alignment
        
        self.host_sat = QComboBox()
        self.target_sat = QComboBox()
        
        # Add host and target satellite selection to satellite layout
        satellite_layout.addRow("Host Satellite:", self.host_sat)
        satellite_layout.addRow("Target Satellite:", self.target_sat)
        
        control_layout.addRow(satellite_group)  # Add satellite selection group to control layout
        
        # Analyze Link button
        analyze_button = QPushButton("Analyze Link")
        analyze_button.clicked.connect(self.analyze_link)
        control_layout.addRow(analyze_button)  # Add Analyze Link button

        # # Add buttons for finding distances of interest
        # find_100km_button = QPushButton("Find ~100km Distances")
        # find_100km_button.clicked.connect(lambda: self.find_distances_of_interest(100, 10)) # 100km +/- 10km
        # control_layout.addRow(find_100km_button)

        # find_1000km_button = QPushButton("Find ~1000km Distances")
        # find_1000km_button.clicked.connect(lambda: self.find_distances_of_interest(1000, 100)) # 1000km +/- 100km
        # control_layout.addRow(find_1000km_button)
        
        # Ephemeris update rate and latency controls
        ephemeris_group = QGroupBox("Ephemeris Controls")
        ephemeris_layout = QFormLayout(ephemeris_group)
        
        self.sat_pe = QComboBox()
        # Load satellite configuration
        file_path = os.path.join(self.outputdir, 'pointing_error', 'simulation_parameters.json')
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        satellite_names = list(data['r_index'].keys())
        self.sat_pe.addItems(satellite_names)
        ephemeris_layout.addRow("Select Orbit", self.sat_pe)


        self.ephm_update_rate = QSpinBox()
        self.ephm_update_rate.setRange(1, 100)
        self.ephm_update_rate.setValue(1)
        self.ephm_update_rate.setSuffix(" Sec")
        ephemeris_layout.addRow("Update rate:", self.ephm_update_rate)
        
        self.update_no = QSpinBox()
        self.update_no.setRange(0, 10)
        self.update_no.setValue(0)        
        ephemeris_layout.addRow("No. of Updates:", self.update_no)

        self.prop_dur = QSpinBox()
        self.prop_dur.setRange(0, 60)
        self.prop_dur.setValue(10)
        self.prop_dur.setSuffix(" s")
        ephemeris_layout.addRow("Propagation duration:", self.prop_dur)

        
        
        self.ephm_latency = QSpinBox()
        self.ephm_latency.setRange(0, 10)
        self.ephm_latency.setValue(0)
        self.ephm_latency.setSuffix(" s")
        ephemeris_layout.addRow("Ephemeris Latency:", self.ephm_latency)
        
        self.link_distance = QSpinBox()
        self.link_distance.setRange(50, 10000000)
        self.link_distance.setValue(100)
        self.link_distance.setSuffix(" km")
        ephemeris_layout.addRow('Link Distance:', self.link_distance)

        # Add host and target satellite selection to ephemeris controls
        # ephemeris_layout.addRow("Host Satellite:", self.host_sat)  # Host satellite in ephemeris controls
        # ephemeris_layout.addRow("Target Satellite:", self.target_sat)  # Target satellite in ephemeris controls
        
        control_layout.addRow(ephemeris_group)  # Add ephemeris controls to control layout
        
        # Calculate PE button
        calculate_pe_button = QPushButton("Calculate PE")
        calculate_pe_button.clicked.connect(self.calculate_pe_ephemeris)
        control_layout.addRow(calculate_pe_button)  # Add Calculate PE button
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Setup graphics tab
        graphics_layout = QVBoxLayout(self.link_graphics)
        self.link_plot = QWebEngineView()
        self.link_plot.settings().setAttribute(QWebEngineSettings.WebGLEnabled, True)
        self.link_plot.settings().setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, True)
        self.link_plot.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.link_plot.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        graphics_layout.addWidget(self.link_plot)
        
    def find_distances_of_interest(self, target_distance_km, tolerance_km):
        outputdir = self.run_orbit_simulation(hp_setting=1,sim_time=24*3600, pe_flag = True)
        """Calculate pointing error from ephemeris"""
        if not  self.simulation_data:
            return
        """
        Finds and prints time instances and exact ranges where the link distance
        is approximately equal to the target_distance_km within a given tolerance.
        """
        # if not self.link_data:
        #     print("Error: No link data available. Please run 'Analyze Link' first.")
        #     return
        # tolerance_km = 10
        target_distance_m = target_distance_km * 1000
        tolerance_m = tolerance_km * 1000

        min_range = target_distance_m - tolerance_m
        max_range = target_distance_m + tolerance_m

        # times = self.link_data['time']
        # ranges = self.link_data['range']
        self.simulation_data
        host_chosen = self.host_sat.currentText()
        target_chosen = self.target_sat.currentText()
        # Get parameters
        # update_rate = self.ephm_update_rate.value()
        # latency = self.ephm_latency.value()
        importlib.reload(j2prop)

        
        link_distance_m = self.link_distance.value()*1e3
        
        data_raw, simulation_parameters = dputil.load_constellation_data(full_path = outputdir)

        t_j2000 = data_raw[:,0]
        t_gps = t_j2000 + t_conv.dt_j2000tt2gps()
        r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
        v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
        
        r_target = data_raw[:,simulation_parameters['r_index'][target_chosen]]
        v_target = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]]
        LOS_true = np.empty((0,3))
        LOS_prop = np.empty((0,3))

        LOS_eci = r_target  - r_host
        ranges  = np.linalg.norm(LOS_eci, axis = 1)

        # Find indices where ranges is within the specified tolerance
        mask = (ranges >= min_range) & (ranges <= max_range)
        times = t_j2000

        found_times = times[mask]
        if len(found_times) < 1:
            print(f"\nNo distances found approximately {target_distance_km} km (tolerance +/- {tolerance_km} km).")
            return
        found_ranges = ranges[mask]
        found_host_r = r_host[mask]
        found_host_v = v_host[mask]
        found_target_r = r_target[mask]
        found_target_v = v_target[mask]

        r_0 = found_target_r[0,:]
        v_0 = found_target_v[0,:]
        X = np.hstack((r_0,v_0))
        dt = found_times[1] - found_times[0]
        t_end = found_times[0]+dt*(len(found_times)-1)
        r_target_predicted = j2prop.propagate_orbit(X, t_start=found_times[0], t_end=t_end, t_step=dt)
        r_predicted = r_target_predicted[:,[1,2,3]]
        dr = found_host_r - r_predicted
      
        fig, ax = plt.subplots(1,2)
        fig.suptitle(f"Pointing Error for Link Distance: {_m/1e3:.2f} km")
        ax[0].plot(found_times-found_times[0], np.linalg.norm(dr,axis = 1))
        ax[1].plot(1e-3*found_ranges, np.linalg.norm(dr, axis = 1))
        ax[0].set_ylabel('dr [m]')
        ax[1].set_xlabel('Range [km]')
        ax[0].grid()
        ax[1].grid()
        plt.show()
        
    def select_tle_from_file(self):

        self.getfile()
        self.tle_file = self.data_fname[0]
        #self.sat_selector = tle_file
        return self.tle_file

    def setup_ogs_tab(self):
        """Setup the OGS analysis tab"""
        layout = QVBoxLayout(self.ogs_tab)
        
        # Control panel for Ground Station Visibility
        control_group = QGroupBox("Ground Station Visibility Controls")
        control_layout = QFormLayout()

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        OGS_layout = QHBoxLayout()
        
        # Add refresh button
        refresh_button = QPushButton("Refresh TLE Data")
        refresh_button.clicked.connect(self.refresh_tle_data)
        button_layout.addWidget(refresh_button)
        
        # Add save button
        save_button = QPushButton("Save TLE")
        save_button.clicked.connect(self.save_tle_data)
        button_layout.addWidget(save_button)
        
        # Add the button layout to the form
        control_layout.addRow(button_layout)

        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(line)

        # Dropdown for selecting Ground Station
        self.gs_selector = QComboBox()
        control_layout.addRow("Select Ground Station:", self.gs_selector)

        # Add satellite selection dropdown
        self.sat_selector = QComboBox()
        control_layout.addRow("Select Satellite:", self.sat_selector)

        # Add Select TLE from file button
        tle_file_button = QPushButton("Select TLE from File")
        tle_file_button.clicked.connect(self.select_tle_from_file)
        control_layout.addRow(tle_file_button)
        
        
        # Add TLE information display
        self.tle_info_label = QLabel()
        self.tle_info_label.setWordWrap(True)
        self.tle_info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        control_layout.addRow("TLE Information:", self.tle_info_label)
        
        # Connect the satellite selection signal
        self.sat_selector.currentTextChanged.connect(self.update_tle_info)

        # Load ground station names and populate the dropdown
        try:
            gs_config_path = os.path.join(self.datadir, 'ground_stations.json')
            tle_path = os.path.join(self.datadir, 'sat', 'tle.json')
            
            if not os.path.exists(gs_config_path):
                print(f"Error: Ground station configuration file not found at {gs_config_path}")
                return
            
            with open(gs_config_path, 'r') as f:
                ground_stations = json.load(f)
            
            # Load TLE data
            if os.path.exists(tle_path):
                with open(tle_path, 'r') as f:
                    self.tle_json_data = json.load(f)
            else:
                print(f"Error: TLE file not found at {tle_path}")
                self.tle_json_data = {}
            
            if isinstance(ground_stations, dict):
                gs_names = list(ground_stations.keys())
                self.gs_selector.addItems(gs_names)
                # Store the full ground stations data for later use in calculation
                self.ground_stations_data = ground_stations
            else:
                raise ValueError("Ground stations data is not in the expected format")
                
        except FileNotFoundError:
            print(f"Error: Ground station configuration file not found at {gs_config_path}")
            self.gs_selector.addItem("No Ground Stations Found")
            self.gs_selector.setEnabled(False)
            self.ground_stations_data = {}
        except json.JSONDecodeError as e:
            print(f"Error parsing ground stations JSON: {e}")
            self.gs_selector.addItem("Invalid Ground Stations File")
            self.gs_selector.setEnabled(False)
            self.ground_stations_data = {}
        except Exception as e:
            print(f"Error loading ground stations: {e}")
            self.gs_selector.addItem("Error Loading Ground Stations")
            self.gs_selector.setEnabled(False)
            self.ground_stations_data = {}

        # Load satellite names from API
        try:
            import requests
            # response = requests.get('https://api.keeptrack.space/v2/sats')
            # response = requests.get('https://api.keeptrack.space//v4/catalog/latest')

            url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
            response = requests.get(url)

            if response.status_code == 200:
                # sat_data = response.json()
                sat_data = response.text
                # Create a dictionary to store satellite data
                self.tle_data = {'satellites': {}}
                
                # Process each satellite from the API
                for sat in sat_data:
                    if 'name' in sat and 'tle1' in sat and 'tle2' in sat:
                        sat_name = sat['name']
                        self.tle_data['satellites'][sat_name] = {
                            'name': sat_name,
                            'line1': sat['tle1'],
                            'line2': sat['tle2'],
                            'norad_id': sat.get('norad_id', ''),
                            'description': sat.get('payload', ''),
                            'launch_date': sat.get('launchDate', ''),
                            'country': sat.get('country', '')
                        }
                
                # Add satellite names to the dropdown in alphabetical order
                sat_names = sorted(list(self.tle_data['satellites'].keys()))
                self.sat_selector.addItems(sat_names)
                
            else:
                raise Exception(f"API request failed with status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching satellite data from API: {e}")
            self.sat_selector.addItem("Error Loading Satellites")
            self.sat_selector.setEnabled(False)
            self.tle_data = {}
        except Exception as e:
            print(f"Error processing satellite data: {e}")
            self.sat_selector.addItem("Error Loading Satellites")
            self.sat_selector.setEnabled(False)
            self.tle_data = {}

        # Add a button to trigger the ground station visibility calculation
        calculate_gs_visibility_button = QPushButton("Calculate Ground Station Visibility")
        if  self.sat_selector.setEnabled(False)==False:
            calculate_gs_visibility_button.clicked.connect(lambda: self.calculate_ground_station_visibility(self.sat_selector.currentText()))
        else:
            calculate_gs_visibility_button.clicked.connect(lambda: self.calculate_ground_station_visibility(self.tle_file))
        
        # QDateTimeEdit widget
        self.datetime_edit = QDateTimeEdit()
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss[UTC]")
        self.datetime_edit.setCalendarPopup(True)  # allows calendar selection
        self.datetime_edit.setDateTime(QDateTime.currentDateTimeUtc())

        OGS_ECI_button = QPushButton("OGS in ECI")
        OGS_ECI_button.clicked.connect(lambda: self.ground_station_eci(self.gs_selector.currentText(), self.datetime_edit.dateTime()))
        
        OGS_pointing_button = QPushButton("Calculate OGS Pointing")
        OGS_pointing_button.clicked.connect(self.telescope_ecef_quat)
        
        self.OGS_combo = QComboBox()
        self.OGS_combo.addItems(["Selected Sat","Sun","Moon","Custom"])

        self.frame_combo = QComboBox()
        self.frame_combo.addItems(["NWU","NED","ENU"])

        OGS_layout.addWidget(OGS_ECI_button)
        OGS_layout.addWidget(OGS_pointing_button)
        OGS_layout.addWidget(self.OGS_combo)
        OGS_layout.addWidget(self.frame_combo)


        
        control_layout.addWidget(self.datetime_edit)
        control_layout.addRow(OGS_layout)
        control_layout.addRow(calculate_gs_visibility_button)        

        # Setup graphics tab for OGS
        graphics_layout = QVBoxLayout(self.ogs_graphics)
        self.ogs_plot = QWebEngineView() # Store as instance variable
        self.ogs_plot.settings().setAttribute(QWebEngineSettings.WebGLEnabled, True)
        self.ogs_plot.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        graphics_layout.addWidget(self.ogs_plot)

        self.gs_visibility_stats_label = QLabel()
        self.gs_visibility_stats_label.setWordWrap(True)
        self.gs_visibility_stats_label.setStyleSheet("QLabel { background-color: #e8f4ff; padding: 6px; border-radius: 4px; }")
        control_layout.addRow(self.gs_visibility_stats_label)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

    def update_tle_info(self, selected_satellite):
        """Update the TLE information display when a satellite is selected"""
        if not selected_satellite or selected_satellite == "Error Loading Satellites":
            self.tle_info_label.setText("No satellite selected or error loading satellites")
            return None

        # Try to get information from the TLE JSON file first
        tle_info = ""
        tle_data = None
        if hasattr(self, 'tle_json_data') and selected_satellite in self.tle_json_data:
            sat_data = self.tle_json_data[selected_satellite]
            tle_info = f"""
            <b>Satellite Information from TLE File:</b><br>
            Name: {selected_satellite}<br>
            Description: {sat_data.get('description', 'N/A')}<br>
            Launch Date: {sat_data.get('launch_date', 'N/A')}<br>
            Country: {sat_data.get('country', 'N/A')}<br>
            TLE Line 1: {sat_data.get('line1', 'N/A')}<br>
            TLE Line 2: {sat_data.get('line2', 'N/A')}<br>
            """
            tle_data = sat_data
        # If not found in TLE JSON, try to get from API data
        elif hasattr(self, 'tle_data') and selected_satellite in self.tle_data['satellites']:
            sat_data = self.tle_data['satellites'][selected_satellite]
            tle_info = f"""
            <b>Satellite Information from API:</b><br>
            Name: {selected_satellite}<br>
            Description: {sat_data.get('description', 'N/A')}<br>
            Launch Date: {sat_data.get('launch_date', 'N/A')}<br>
            Country: {sat_data.get('country', 'N/A')}<br>
            TLE Line 1: {sat_data.get('line1', 'N/A')}<br>
            TLE Line 2: {sat_data.get('line2', 'N/A')}<br>
            """
            tle_data = sat_data
        else:
            tle_info = f"No TLE information available for {selected_satellite}"

        self.tle_info_label.setText(tle_info)
        return tle_data

    def update_quaternion_integrator(self, text):
        VALID_INTEGRATORS = {'RK4', 'Rapid 4th-Order'}
        self.active_integrator = text if text in VALID_INTEGRATORS else "RK4"
        print(f"Active quaternion integrator: {self.active_integrator}")
    
    def update_quaternion_interpolator(self, text):
        self.active_interpolator = text if text != "SLERP" else "SLERP"
        print(f"Active quaternion integrator: {self.active_interpolator}")

    def read_single_tle(self, file_path):
        """
        Reads a TLE file with exactly 2 lines of TLE data for one satellite.        
        Args:
            file_path (str): Path to the TLE file.            
        Returns:
            dict: A dictionary containing the two lines of TLE data.
        """
        try:
            with open(file_path, 'r') as file:
                # Read exactly two lines from the file
                lines = file.readlines()

                if len(lines) != 2:
                    raise ValueError("The file must contain exactly two lines of TLE data.")
                
                # Strip whitespace/newline characters
                tle_line1 = lines[0].strip()  # First line of TLE (orbital elements)
                tle_line2 = lines[1].strip()  # Second line of TLE (orbital elements)
                
                # Return TLE data as a dictionary
                return tle_line1, tle_line2 
        
        except Exception as e:
            print(f"Error reading TLE file: {e}")
            return None

    def propagate_tle_to_gcrs_gps(self,t_start_iso: str, duration_s: float = 7200, step_s: int = 300, sat: Satrec = None): 
        """
        Propagates ISS from t_start_iso (UTC) and returns DataFrame with GPS seconds.
        """
        # Reference GPS time at J2000 (2000-01-01 12:00:00 TAI = GPS epoch + 19s leap)
        # But astropy handles GPS scale correctly
        # t_ref_gps = Time('2000-01-01T12:00:00', scale='tai')  # GPS zero point

        # Propagation times
        t0_utc = Time(t_start_iso, scale='utc')
        times_utc = t0_utc + np.arange(0, duration_s + step_s, step_s) * u.s

        # GPS time in seconds since GPS epoch
        times_gps = times_utc.gps    # This is the correct GPS seconds (float)
        t_gps_0 = times_gps[0]       # First epoch in GPS seconds

        N = len(times_utc)
        r_gcrs = np.zeros((N, 3))
        v_gcrs = np.zeros((N, 3))

        print(f"Propagating {N} points from {t0_utc.iso} (GPS week second {t_gps_0:.1f})")

        for i, t in tqdm (enumerate(times_utc),total=len(times_utc), desc="Propagating TLE → GCRS(ECI)"):
            
            err, r_teme, v_teme = sat.sgp4(t.jd1, t.jd2)    # Propagating TLE in TEME frame
            if err != 0:
                raise RuntimeError(f"SGP4 propagation error {err} at time {t.iso}")
            pos = CartesianRepresentation(r_teme * u.km)
            vel = CartesianDifferential(v_teme * u.km / u.s)
            teme_state = TEME(pos.with_differentials(vel), obstime=t)

            gcrs_state = teme_state.transform_to(GCRS(obstime=t))

            r_gcrs[i] = gcrs_state.cartesian.xyz.to(u.km).value
            v_gcrs[i] = gcrs_state.velocity.d_xyz.to(u.km/u.s).value

        # Build final array: [t_gps_s, rx, ry, rz, vx, vy, vz]
        propagated_orbit_eci = np.column_stack([
            times_gps,      # GPS seconds (float)
            r_gcrs,         # km
            v_gcrs          # km/s
        ])

        # Create DataFrame
        df = pd.DataFrame(
            data=propagated_orbit_eci,
            columns=['t_gps_s', 'r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z']
        )

        return df

    def calculate_ground_station_visibility(self, selected_satellite, flag=False):
        """Calculate and plot ground station visibility for all satellites."""

        if not selected_satellite:        
            QMessageBox.warning(self, "Warning", "Please select a satellite first.")
            return
        
        if isinstance(selected_satellite, str) and os.path.exists(selected_satellite):        
            # You can customize this depending on file format               
            if selected_satellite.endswith(".txt"):
                QMessageBox.warning(self, "Warning", "Loading TLE from file.")
                tle_line1, tle_line2 = self.read_single_tle(selected_satellite)  
                sat_name = 'Custom'             

            else:
                raise ValueError("Unsupported file format")

        # -------------------------
        # Case 2: Already a DataFrame
        # -------------------------
        elif isinstance(selected_satellite, pd.DataFrame):
            sat_data = selected_satellite

        # -------------------------
        # Case 3: Custom object (your satellite class)
        # -------------------------
        else:
            # assume it's already usable (or adapt this)
            sat_data = selected_satellite
                    # Get TLE data for the selected satellite
            tle_data = self.update_tle_info(selected_satellite)
                    # Get infotmation from the TLE
            tle_line1 = tle_data.get('line1')
            tle_line2 = tle_data.get('line2')
            sat_name = tle_data.get('name')
        
        
            if not tle_data:
                QMessageBox.warning(self, "Warning", "No TLE data available for the selected satellite.")
                return
            
        start_time = self.datetime_edit.dateTime().toPyDateTime().replace(tzinfo=timezone.utc)
        duration_sec = 7200 # seconds
        dt = 5.0

        # --- Construct list of times for high-resolution CSV export (2 hours) ---
        num_steps = int(duration_sec / dt) + 1
        times = [start_time + timedelta(seconds=i*dt) for i in range(num_steps)]

        # === TLE Propagation to CSV (2-hour high-resolution) ===
        propagation_duration_hours = 2
        self.sat = Satrec.twoline2rv(tle_line1, tle_line2)
        t0_utc = Time.now()

        file_name = os.path.join(self.outputdir, 'tables', 'TLE_propagation',
                                 f"propagated_orbit_ECI_{sat_name}_{t0_utc.iso.replace(':', '-')}.csv")

        rows = []
        start = perf_counter()

        # Check if any existing file has the same date
        # Pattern to match files for this satellite
        
        current_date_str = t0_utc.iso.split(" ")[0]
        file_pattern = os.path.join(self.outputdir, 'tables', 'TLE_propagation', f"propagated_orbit_ECI_{sat_name}_*.csv")
        existing_files = glob(file_pattern)
        skip = False
        for f in existing_files:
            base = os.path.basename(f)  # e.g., propagated_orbit_ECI_Custom_2026-04-02 12-12-09.804.csv
            try:
                # Extract timestamp part from filename
                timestamp_part = base.split(f"propagated_orbit_ECI_{sat_name}_")[-1].replace(".csv", "")#"_".join(base.split("_")[-2:]).replace(".csv", "")  # '2026-04-02 12-12-09.804'                
                file_date_str = timestamp_part.split(" ")[0]  # '2026-04-02'
                if file_date_str == current_date_str:
                    skip = True
                    print(f"Skipping: file for date {file_date_str} already exists -> {f}")
                    break
            except IndexError:
                continue

        if not skip:
            for t in tqdm(times, total=len(times), desc="Propagating TLE (2h high-res) in ECI frame"):
                astropy_time = Time(t)
                times_gps_s, r_eci, v_eci = self.propagate_tle(tle_line1, tle_line2, astropy_time, output_frame="ECI")
                rows.append([times_gps_s, *r_eci, *v_eci])

            print(f"Finished 2-hour propagation in {perf_counter() - start:.2f} seconds")

            with open(file_name, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t_gps_s", "r_x", "r_y", "r_z", "v_x", "v_y", "v_z"])
                writer.writerows(rows)

        # === Fast Visibility Analysis over multiple days (9 days) ===
        # Note: We do NOT overwrite 'times' here. Use a different variable for clarity.
        time_step = 90 * u.s
        time_span = 9 * u.day

        vis_times = t0_utc + np.linspace(0, time_span.to_value(u.s),
                                         int(time_span / time_step) + 1) * u.s

        N = len(vis_times)

        print(f"Computing visibility over {time_span} ({N:,} points @ {time_step})...")

        # Get selected ground station
        selected_gs = self.gs_selector.currentText()
        if not selected_gs or selected_gs == "No Ground Stations Found":
            QMessageBox.warning(self, "Warning", "Please select a ground station first.")
            return

        self.gs_data = self.ground_stations_data.get(selected_gs)
        if not self.gs_data:
            QMessageBox.warning(self, "Warning", f"No data available for ground station {selected_gs}")
            return

        # Ground station location
        gs_location = EarthLocation.from_geodetic(
            lat=self.gs_data['latitude_deg'],
            lon=self.gs_data['longitude_deg'],
            height=self.gs_data['altitude_km']
        )

        # Propagate once using SGP4 array
        err, r_teme_km, v_teme_km_s = self.sat.sgp4_array(vis_times.jd1, vis_times.jd2)

        if np.any(err != 0):
            print("Warning: Some SGP4 propagation errors occurred")

        r_teme_m = r_teme_km * 1000.0

        elevation_angles = np.zeros(N)
        ground_track = np.empty((N, 2))
        
        print(f"Processing {N:,} points ... ", end="", flush=True)
        start = perf_counter()

        # Compute GS az/el & ground track 
        for i, t in enumerate(tqdm(vis_times, desc="Computing elevation & ground track")):
            pos = r_teme_m[i] * u.m
            gcrs = GCRS(pos, obstime=t, representation_type='cartesian')
            altaz = gcrs.transform_to(AltAz(location=gs_location, obstime=t))
            elevation_angles[i] = altaz.alt.deg

            itrs = gcrs.transform_to(ITRS(obstime=t))
            ground_track[i, 0] = itrs.spherical.lon.wrap_at(180*u.deg).deg
            ground_track[i, 1] = itrs.spherical.lat.deg

        print(f"Finished in {perf_counter() - start:.2f} seconds")

        # Compute gimbal angles for your gimbaled payload
        gimbal_az, gimbal_el = self.compute_gimbal_angles(selected_gs, vis_times, r_teme_km*1e3, v_teme_km_s*1e3)
        #print(gimbal_az, gimbal_el)

        # === Detect passes (min elevation = 20°) ===
        print(f"Detecting passes (min elevation = 20°) ...")
        min_elevation = 20.0

        rising = (elevation_angles[:-1] < min_elevation) & (elevation_angles[1:] >= min_elevation)
        falling = (elevation_angles[:-1] >= min_elevation) & (elevation_angles[1:] < min_elevation)

        rise_idx = np.where(rising)[0]
        fall_idx = np.where(falling)[0]

        passes = []
        pass_starts = []
        pass_ends = []
        pass_durations = []

        for i, start_idx in enumerate(rise_idx):
            if i >= len(fall_idx):
                break
            end_idx = fall_idx[i]
            if end_idx <= start_idx:
                continue

            duration_min = (vis_times[end_idx] - vis_times[start_idx]).to(u.minute).value
            if duration_min < 0.5:
                continue

            max_elev = np.max(elevation_angles[start_idx:end_idx+1])

            passes.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': vis_times[start_idx],
                'end_time': vis_times[end_idx],
                'duration_min': duration_min,
                'max_elev': max_elev
            })

            pass_starts.append(vis_times[start_idx])
            pass_ends.append(vis_times[end_idx])
            pass_durations.append(duration_min)

        # Statistics
        average_duration = np.mean(pass_durations) if pass_durations else 0.0
        total_days = (vis_times[-1] - vis_times[0]).to(u.day).value
        average_passes_per_day = len(passes) / total_days if total_days > 0 else 0

        print(f"Found {len(passes)} visible passes (≥ {min_elevation}°) in {total_days:.1f} days")
        print(f"Avg duration: {average_duration:.1f} min | {average_passes_per_day:.2f} passes per day\n")      
        print(f"Vectorized coordinate transform for {N:,} points...")
        
        start = perf_counter()

        cart         = CartesianRepresentation(r_teme_m[:, 0]*u.m, r_teme_m[:, 1]*u.m, r_teme_m[:, 2]*u.m)
        gcrs_arr     = GCRS(cart, obstime=vis_times, representation_type='cartesian')
        altaz_arr    = gcrs_arr.transform_to(AltAz(location=gs_location, obstime=vis_times))

        elevation_angles = altaz_arr.alt.deg          # shape (N,)
        azimuth_angles   = altaz_arr.az.deg           # shape (N,)  ← new

        itrs_arr           = gcrs_arr.transform_to(ITRS(obstime=vis_times))
        ground_track[:, 0] = itrs_arr.spherical.lon.wrap_at(180 * u.deg).deg
        ground_track[:, 1] = itrs_arr.spherical.lat.deg

        print(f"Vectorized transform done in {perf_counter() - start:.2f} s")
        
        self.plot_ground_station_visibility(selected_gs, selected_satellite,
                                   pass_durations, pass_starts, pass_ends,
                                   ground_track, vis_times,gimbal_az ,gimbal_el,
                                   elevation_angles, azimuth_angles, passes,
                                   tle_line1, tle_line2)

        stats_text = (
            f"<b>Short propagation (CSV):</b> {propagation_duration_hours:.2f} hours <br>"
            f"<b>Visibility analysis:</b> {time_span} <br>"
            f"<b>Average pass duration:</b> {60*average_duration:.2f} seconds<br>"
            f"<b>Average passes per day:</b> {average_passes_per_day:.2f}"
        )
        self.gs_visibility_stats_label.setText(stats_text)

        # --- Add Cesium 3D groundstation plot ---
        self.plot_3d_groundstation(
            ground_track=ground_track,
            passes =passes,                        
            selected_gs=selected_gs,
            satellite = selected_satellite
        )

    def refresh_tle_data(self):
        """Refresh TLE data from the API"""
        try:
            import requests
            response = requests.get('https://api.keeptrack.space/v2/sats')
            if response.status_code == 200:
                sat_data = response.json()
                # Clear existing data
                self.tle_data = {'satellites': {}}
                self.sat_selector.clear()
                
                # Process each satellite from the API
                for sat in sat_data:
                    if 'name' in sat and 'tle1' in sat and 'tle2' in sat:
                        sat_name = sat['name']
                        self.tle_data['satellites'][sat_name] = {
                            'name': sat_name,
                            'line1': sat['tle1'],
                            'line2': sat['tle2'],
                            'norad_id': sat.get('norad_id', ''),
                            'description': sat.get('payload', ''),
                            'launch_date': sat.get('launchDate', ''),
                            'country': sat.get('country', '')
                        }
                
                # Add satellite names to the dropdown in alphabetical order
                sat_names = sorted(list(self.tle_data['satellites'].keys()))
                self.sat_selector.addItems(sat_names)
                
                # Update the TLE JSON data attribute
                self.tle_json_data = self.tle_data['satellites']
                
                # Update the display for the currently selected satellite
                current_satellite = self.sat_selector.currentText()
                if current_satellite:
                    _= self.update_tle_info(current_satellite)
                
                print("TLE data refreshed successfully")
                QMessageBox.information(self, "Success", "TLE data has been refreshed successfully!")
            else:
                raise Exception(f"API request failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error refreshing TLE data: {e}")
            QMessageBox.warning(self, "Error", f"Failed to refresh TLE data: {str(e)}")

    def save_tle_data(self):
        """Save the current TLE data to the local JSON file"""
        try:
            if not hasattr(self, 'tle_data') or not self.tle_data['satellites']:
                QMessageBox.warning(self, "Warning", "No TLE data available to save. Please refresh the data first.")
                return

            tle_path = os.path.join(self.datadir, 'sat', 'tle.json')
            os.makedirs(os.path.dirname(tle_path), exist_ok=True)
            
            with open(tle_path, 'w') as f:
                json.dump(self.tle_data['satellites'], f, indent=4)
            
            print("TLE data saved successfully")
            QMessageBox.information(self, "Success", "TLE data has been locally saved successfully!")
            
        except Exception as e:
            print(f"Error saving TLE data: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save TLE data: {str(e)}")

    def setup_moon_tab(self):
        """Setup the new tab"""
        layout = QVBoxLayout(self.moon_phase_tab)
        label = QLabel("Sun/Moon Phase Calculation!\n Default Geo-location: Munich,DE")
        layout.addWidget(label)

        # --- Geo-location input ---
        txt_label = QLabel("Geo-location [lat, lon, alt[m]]:")
        layout.addWidget(txt_label)
        geo_layout = QHBoxLayout()
        self.geo_line_edit = QLineEdit("[48.137017, 11.419067, 567.5]")  # default value
        geo_layout.addWidget(txt_label)
        geo_layout.addWidget(self.geo_line_edit)

        geo_get_button = QPushButton("Get")
        geo_get_button.clicked.connect(self.save_geo_location)
        geo_layout.addWidget(geo_get_button)

        layout.addLayout(geo_layout)

        control_group = QGroupBox("Sun/Moon Postion Calculation")
        control_layout = QFormLayout()
        
        # Dropdown (QComboBox)
        self.celes_item = QComboBox()
        self.celes_item.addItems(["Sun", "Moon"])  # Add items to dropdown

        # Label to show selection
        label = QLabel("Select Celestial Body")
        sun_button_layout = QHBoxLayout()

        sun_phase_button = QPushButton("Phase")
        sun_phase_button.clicked.connect(self.run_phase_calculation)
        sun_button_layout.addWidget(sun_phase_button)

        self.sun_full_button = QPushButton("GS-Celestial Body[ECI]-CONDOR")
        self.sun_full_button.clicked.connect(self.sun_eci_full) # (lambda: setattr(self, 'was_clicked', True))
        sun_button_layout.addWidget(self.sun_full_button)

        control_layout.addWidget(label)
        control_layout.addWidget(self.celes_item)
        control_layout.addRow(sun_button_layout)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Add date input widgets
        date_group = QGroupBox("Date Range")
        date_layout = QFormLayout(date_group)

        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(datetime.now())#(2025, 6, 11).date()) # Default start date
        date_layout.addRow("Start Date:", self.start_date_edit)

        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)        
        # self.end_date_edit.setDate(datetime.now())#(2025, 6, 30).date()) # Default end date
        self.end_date_edit.setDate(QDate.currentDate().addDays(1))
        date_layout.addRow("End Date:", self.end_date_edit)

        layout.addWidget(date_group)

        # Setup graphics tab for new tab
        graphics_layout = QVBoxLayout(self.moon_graphics)
        self.moon_plot_widget = QWebEngineView() # Store as instance variable
        self.moon_plot_widget.settings().setAttribute(QWebEngineSettings.WebGLEnabled, True)
        self.moon_plot_widget.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        graphics_layout.addWidget(self.moon_plot_widget)
        
    def save_geo_location(self):
        text = self.geo_line_edit.text().strip()
        try:
            # Remove brackets if present
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]

            # Split by comma
            parts = [p.strip() for p in text.split(",")]
            if len(parts) != 3:
                raise ValueError

            # Convert to float
            geo_value = [float(p) for p in parts]

            # Save to instance variable
            self.ogs_geo_location = geo_value
            print("Geo-location saved:", self.ogs_geo_location)
            QMessageBox.information(self, "Success", f"Geo-location saved: {geo_value}")

        except Exception as e:
            print("Error:", e)
            QMessageBox.warning(self, "Invalid Input", 
                                "Please enter a valid list of [lat, lon, alt], e.g., [48.137017, 11.419067, 567.5]")

    def load_and_populate_sat_names(self, sim_folder):
        
        data_raw, simulation_parameters = dputil.load_constellation_data(full_path=sim_folder)
        if simulation_parameters is not None:
            sat_names = simulation_parameters['sat_names']
            self.host_sat.clear()
            self.target_sat.clear()
            self.host_sat.addItems(sat_names)
            self.target_sat.addItems(sat_names)
            print(f'Satellites found: {sat_names}')
            self.simulation_data['sat_names'] = sat_names
        else:
            print("Failed to load satellite names from simulation_parameters.json")

    def plot_trajectories(self):
        """Plot 3D trajectories of all satellites"""
        if not self.simulation_data:
            return
        
        states_array = self.simulation_data['states']
        sat_names = self.simulation_data['sat_names']
        n_sats = len(sat_names)
        
        # Create subplots for x, y, z positions
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('X Position', 'Y Position', 'Z Position'))
        
        # Plot each satellite's position components
        for ii in range(n_sats):
            # Get position data
            x = states_array[:, 6*ii+1]
            y = states_array[:, 6*ii+2]
            z = states_array[:, 6*ii+3]
            
            # Add X position
            fig.add_trace(
                go.Scatter(x=self.simulation_data['time'],
                          y=x,
                          name=f'{sat_names[ii]} X'),
                row=1, col=1
            )
            
            # Add Y position
            fig.add_trace(
                go.Scatter(x=self.simulation_data['time'],
                          y=y,
                          name=f'{sat_names[ii]} Y'),
                row=2, col=1
            )
            
            # Add Z position
            fig.add_trace(
                go.Scatter(x=self.simulation_data['time'],
                          y=z,
                          name=f'{sat_names[ii]} Z'),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True)
        fig.update_yaxes(title_text="Position [m]", row=1, col=1)
        fig.update_yaxes(title_text="Position [m]", row=2, col=1)
        fig.update_yaxes(title_text="Position [m]", row=3, col=1)
        fig.update_xaxes(title_text="Time [s]", row=3, col=1)
        
        # Convert to HTML and display
        html = fig.to_html(include_plotlyjs='cdn')
        self.orbit_plot.setHtml(html, QUrl('about:blank'))

    def plot_3d_trajectories(self):
        if not self.simulation_data:
            print("Error: No simulation data available for 3D trajectory plotting.")
            return
        
        states_array = self.simulation_data['states']
        sat_names = self.simulation_data['sat_names']
        n_sats = len(sat_names)
        times = states_array[:, 0]
        # elevation_angles = self.link_data['elevation']

        # Convert ECI to ECEF for each satellite
        ecef_positions = []  # List of (xs, ys, zs) for each satellite
        for ii in range(n_sats):
            xs = states_array[:, 6*ii+1]
            ys = states_array[:, 6*ii+2]
            zs = states_array[:, 6*ii+3]
            ecef_xs, ecef_ys, ecef_zs = self.eci_to_ecef(times, xs, ys, zs)
            

            r_host = states_array[:, 1+6*ii:4+6*ii]
            v_host = states_array[:, 4+6*ii:7+6*ii]

            #r_target = states_array[:, 1+6*(ii+1):4+6*(ii+1)]
            # self.tudat_converter.rot_eci2ecef(X_eci)
            ecef_positions.append((ecef_xs, ecef_ys, ecef_zs))

            # Calculate RSW to ECI rotation matrix
            # tudconv = tudatconv.tudat_predictor()
            # ROT_RSWfromECI = np.array([tudconv.calc_rotrsweci(r_h=r_host[ii,:], v_h=v_host[ii,:]) 
            #                         for ii, r in enumerate(r_host)])
            
            # # Convert to quaternions
            # quat_eci2rsw = np.array([att_conv.convert_dcm2quat(dcm_ii) 
            #                         for dcm_ii in ROT_RSWfromECI])
            
            # # Calculate azimuth, elevation, and range
            # aer = ae_calc.calc_ae_full(r_host, r_target, attitude_eci2bf=quat_eci2rsw,
            #                         check_occultation=0)  # rad rad m

        # Convert states_array to a list for JSON serialization (now using ECEF)
        # We'll build a new list of shape (n_times, n_sats*3) for Cesium
        ecef_array_list = []
        n_times = len(times)
        for j in range(n_times):
            row = []
            for ii in range(n_sats):
                row.extend([float(ecef_positions[ii][0][j]), float(ecef_positions[ii][1][j]), float(ecef_positions[ii][2][j])])
            ecef_array_list.append(row)

        # Prepare ISO8601 time strings for Cesium animation
        
        times_iso = []
        for t in times:
            # Convert UNIX seconds to ISO8601 string (UTC)
            dt = datetime.utcfromtimestamp(t).replace(tzinfo=timezone.utc)
            times_iso.append(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
        
        ## load cesium html
        # Read template file
        template_path = Path(self.datadir)/'cesium_3D_trajectory_copy.html' 
        template = template_path.read_text()

        # Import html file for cesium
        cesium_html = template.format(
            n_sats=n_sats,
            n_times=n_times,
            sat_names=json.dumps(sat_names),
            ecef_states=json.dumps(ecef_array_list),
            times_iso=json.dumps(times_iso)
        )
        ## end cesium html
        
        # Save the generated HTML to the figures folder
        output_dir = os.path.join(self.outputdir, 'figures')
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, 'Trajectories_3D.html')
        with open(output_file_path, 'w') as f:
            f.write(cesium_html)
        print(f"3D trajectory visualization saved to {output_file_path}.")
        # Debug: Print min/max position values for each satellite (ECEF)
        for ii in range(n_sats):
            ecef_xs, ecef_ys, ecef_zs = ecef_positions[ii]
            #print(f"Satellite {sat_names[ii]} (ECEF): X range [{{ecef_xs.min()}}, {{ecef_xs.max()}}], Y range [{{ecef_ys.min()}}, {{ecef_ys.max()}}], Z range [{{ecef_zs.min()}}, {{ecef_zs.max()}}]")
            print(f"Satellite {sat_names[ii]} (ECEF): X range [{ecef_xs.min()}, {ecef_xs.max()}], Y range [{ecef_ys.min()}, {ecef_ys.max()}], Z range [{ecef_zs.min()}, {ecef_zs.max()}]")

        self.start_http_server_and_open(output_file_path)
        # self.start_http_server_and_open(output_file_path)

    def start_http_server_and_open(self, html_file_path):
        """
        Start a Python HTTP server from the project root and open the given HTML file in the default browser.
        Also checks if the .glb file is accessible and prints debug info.
        If port 8000 is in use, automatically finds and kills the process before starting the server.
        The server will automatically shut down and release the port when the browser tab is closed.
        """
        import threading
        import webbrowser
        import http.server
        import socketserver
        import time
        import os
        import requests
        import subprocess
        import sys

        # Find the project root (assume this script is run from anywhere in the project)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        os.chdir(project_root)
        port = 8000
        server_holder = {}

        class StoppableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True
            def shutdown_server(self):
                self.shutdown()
                self.server_close()

        def run_server():
            Handler = http.server.SimpleHTTPRequestHandler
            with StoppableTCPServer(("", port), Handler) as httpd:
                server_holder['server'] = httpd
                print(f"Serving HTTP at http://localhost:{port}/ ... (Ctrl+C to stop)")
                httpd.serve_forever()

        def is_port_in_use(port):
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        def kill_process_on_port(port):
            try:
                result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
                pids = result.stdout.strip().split('\n')
                if pids and pids[0]:
                    for pid in pids:
                        print(f"[INFO] Killing process {pid} on port {port}...")
                        subprocess.run(["kill", "-9", pid])
                    return True
                else:
                    print(f"[INFO] No process found on port {port} to kill.")
                    return False
            except Exception as e:
                print(f"[ERROR] Could not kill process on port {port}: {e}")
                return False

        if is_port_in_use(port):
            print(f"[WARNING] Port {port} already in use. Attempting to kill the process...")
            killed = kill_process_on_port(port)
            if killed:
                time.sleep(2)  # Give the OS a moment to release the port
        # After killing, check again
        if not is_port_in_use(port):
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            time.sleep(1)  # Give the server a moment to start
        else:
            print(f"[ERROR] Port {port} is still in use. Please free it manually.")

        # Open the HTML file in the default browser and get the browser process if possible
        rel_path = os.path.relpath(html_file_path, project_root)
        port1 = 8001
        url = f"http://localhost:{port}/{rel_path}"
        print(f"[INFO] Attempting to open {url} in web browser...")
        browser_proc = None
        try:
            # Try to use webbrowser.get() to launch a new browser process
            try:
                browser = webbrowser.get()
                browser_proc = subprocess.Popen([browser.name, url]) if hasattr(browser, 'name') else None
            except Exception:
                # Fallback to webbrowser.open (may not give us a process handle)
                opened = webbrowser.open(url)
                if not opened:
                    print(f"[WARNING] webbrowser.open() did not report success. Please copy and paste this URL into your browser:")
                    print(url)
        except Exception as e:
            print(f"[ERROR] Could not open browser automatically: {e}")
            print(f"Please copy and paste this URL into your browser:")
            print(url)

        # Monitor the browser process and shut down the server when the tab/window is closed
        if browser_proc is not None:
            print("[INFO] Monitoring browser process. The server will shut down when you close the browser tab/window.")
            try:
                browser_proc.wait()
            except KeyboardInterrupt:
                print("[INFO] KeyboardInterrupt received. Shutting down server.")
            finally:
                if 'server' in server_holder and server_holder['server']:
                    print("[INFO] Shutting down HTTP server...")
                    server_holder['server'].shutdown_server()
                    print("[INFO] Server shut down and port released.")
        else:
            print("[INFO] Could not monitor browser process. The server will continue running in the background.")

        # Check if the .glb file is accessible via HTTP
        glb_url = f"http://localhost:{port}/examples/resources/sat/ICESat-2.glb"
        try:
            resp = requests.get(glb_url, timeout=5)
            if resp.status_code == 200:
                print(f"[INFO] .glb file is accessible at {glb_url}")
            else:
                print(f"[WARNING] .glb file returned status code {resp.status_code} at {glb_url}")
        except Exception as e:
            print(f"[ERROR] Could not access .glb file at {glb_url}: {e}")
            print("If the satellite model is not visible, check the server path and permissions.")

    def calculate_pe(self, active_interpolator, respect_update_rate: bool = False):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.animation import FuncAnimation
        import quaternion_slerp_squad as quat_squad

        
        """Calculate pointing error for all update_rate/latency combos, plot only selected, save all CSVs.
        Default behavior (respect_update_rate=False) preserves original dense 5 ms outputs.
        If respect_update_rate=True, the function also generates per-update-rate evaluation CSVs
        with filenames suffixed by '_respectrate.csv'.
        """
        active_interpolator  = self.active_interpolator
        print(f"Interpolation method:{active_interpolator}")
        if not self.attitude_data:
            return
        
        if self.rsepect_rate_flag:
            respect_update_rate = True

        selected_update_rate = int(self.update_rate.currentData())
        selected_latency = int(self.latency.currentData())
        update_rates = [1, 2, 5, 10]
        latencies = [0, 1, 2, 3, 4]
        selected_result = None
        selected_swapped_result = None

        # Get attitude profile info for filename
        settings_name = self.settings_combo.currentText()
        roll = float(self.roll.value())
        pitch = float(self.pitch.value())
        yaw = float(self.yaw.value())

        dt_req = 5e-3  # 5 ms (original dense sampling)
        n_digits = 3
        propagators_enabled = 1

        # Common preprocessing (do once)
        t_vec = np.round(self.attitude_data['time'], n_digits)
        q_true = np.column_stack((self.attitude_data['quaternions'], self.attitude_data['quaternion_rates']))
        t_req = np.round(np.arange(t_vec[0], t_vec[-1] + dt_req, dt_req), n_digits)
        t_gps_interp = CubicSpline(t_vec, t_vec, axis=0)
        q_host_interp = CubicSpline(t_vec, q_true, axis=0)
        #q_host_interp = quat_squad.make_cubic_spline_interpolator(t_vec, self.attitude_data['quaternions'], self.attitude_data['quaternion_rates'])
         

        def process_quaternion_prediction(apply_sign_swap=False, do_respect_rate=False):
            """
            Internal worker that produces one set of CSVs for either:
                do_respect_rate == False -> dense 5 ms sampling
                do_respect_rate == True  -> sampling at 1/update_rate
            Returns result_for_selected (for plotting) if the processed combo matches selected UI combo.
            """
            result_for_selected = None
            if self.rsepect_rate_flag:
                do_respect_rate = True

            for update_rate in update_rates:
                for latency in latencies:
                    # decide timestep for evaluation
                    dt_eval = (1.0 / update_rate) if do_respect_rate else dt_req
                    t_eval = np.round(np.arange(t_vec[0], t_vec[-1] + dt_eval, dt_eval), n_digits)

                    # file & folder
                    output_dir = os.path.join(self.outputdir, 'tables', f'{settings_name}_quatpred')
                    os.makedirs(output_dir, exist_ok=True)

                    suffix = '_respectrate' if do_respect_rate else ''
                    filename_prefix = 'swapped_true_quat' if apply_sign_swap else 'true_quat'
                    output_file = os.path.join(
                        output_dir,
                        f'{filename_prefix}{settings_name}_roll{roll}_pitch{pitch}_yaw{yaw}_{update_rate}Hz_{latency}s_{active_interpolator}{suffix}.csv'
                    )

                    if os.path.exists(output_file):
                        print(f'File exists: {output_file}')
                        if update_rate == selected_update_rate and latency == selected_latency:
                            result_for_selected = self._load_pe_csv(output_file)
                        continue

                    # timing + latency
                    dt_gap_att_h = np.round(1 / update_rate, 3)
                    dt_latency = dt_gap_att_h * latency
                    t_update_arrival = np.round(np.arange(t_vec[0], t_vec[-1] + dt_gap_att_h, dt_gap_att_h), 3)

                    # evaluate ground truth / predicted times at t_eval
                    t_gps_interp_eval = t_gps_interp(t_eval)
                    q_host_true_eval = q_host_interp(t_eval)

                    t_gps_pred_eval = np.zeros_like(t_gps_interp_eval)
                    q_host_pred_eval = np.zeros_like(q_host_true_eval)
                    t_stamps_updates = t_update_arrival - dt_latency

                    # optional quaternion sign swap (keeps original logic)
                    if apply_sign_swap:
                        index = np.where(t_gps_interp_eval == 10)[0]
                        if len(index) > 0:
                            q_host_true_eval[index[0]:] = -q_host_true_eval[index[0]:]

                    # data used for held/propagated attitude
                    data_full_att_h = q_host_interp(t_stamps_updates)
                    ii_next_att_h = 0
                    quat_interp = interp.we_interpolating()

                    # iterate through evaluation times and compute predicted quat
                    for ii, t_ii in enumerate(t_gps_interp_eval):
                        # advance held index if this eval time reached the next update arrival
                        if ii_next_att_h < len(t_update_arrival) and t_ii >= t_update_arrival[ii_next_att_h] and t_ii < t_update_arrival[-1]:
                            data_att_h = data_full_att_h[ii_next_att_h]
                            data_att_h_held = data_att_h  # for propagator off
                            if ii_next_att_h >= 1:
                                try:
                                    quat_interp.get_quad_interpolant(
                                        t_both=t_stamps_updates[ii_next_att_h-1:ii_next_att_h+1],
                                        r_both=data_full_att_h[ii_next_att_h-1:ii_next_att_h+1, :4],
                                        v_both=(data_full_att_h[ii_next_att_h-1:ii_next_att_h+1, 4:]
                                                if data_full_att_h.shape[1] > 4 else np.zeros((2, 4))),
                                    )
                                except Exception as e:
                                    print(f"Interpolation error: {e}")
                            ii_next_att_h += 1

                        if ii_next_att_h >= 2:
                            if propagators_enabled:
                                data_att_interp = quat_interp.interpolate_flexible(t_ii)
                            else:
                                data_att_interp = data_att_h_held[:4]
                        else:
                            data_att_interp = [0, 0, 0, 0]

                        q_host_pred_eval[ii, :4] = data_att_interp
                        t_gps_pred_eval[ii] = t_ii

                    # compute pointing error using existing att_pred helper
                    pe_remaining = [
                        att_pred.eval_pred_error(q_pred_ii, q_true_ii)[1]
                        for q_pred_ii, q_true_ii in zip(q_host_true_eval[:, :4], q_host_pred_eval[:, :4])
                    ]
                    pe_remaining = np.array(pe_remaining)

                    # save dataframe
                    # note: maintain same columns as original code
                    df_dict = {
                        'time': t_gps_interp_eval,
                        'pe': pe_remaining,
                        'q_true_w': q_host_true_eval[:, 0],
                        'q_true_x': q_host_true_eval[:, 1],
                        'q_true_y': q_host_true_eval[:, 2],
                        'q_true_z': q_host_true_eval[:, 3],
                        'q_true_w_dot': q_host_true_eval[:, 4],
                        'q_true_x_dot': q_host_true_eval[:, 5],
                        'q_true_y_dot': q_host_true_eval[:, 6],
                        'q_true_z_dot': q_host_true_eval[:, 7],
                        'q_pred_w': q_host_pred_eval[:, 0],
                        'q_pred_x': q_host_pred_eval[:, 1],
                        'q_pred_y': q_host_pred_eval[:, 2],
                        'q_pred_z': q_host_pred_eval[:, 3],
                    }
                    pd.DataFrame(df_dict).to_csv(output_file, index=False)
                    swap_msg = 'swapped quaternions ' if apply_sign_swap else ''
                    mode_msg = ' (respect update rate)' if do_respect_rate else ''
                    print(f'Saved {swap_msg}{output_file}{mode_msg}')

                    # if this is the UI-selected combo, capture for visualization
                    if update_rate == selected_update_rate and latency == selected_latency:
                        result_for_selected = {
                            'time': t_gps_interp_eval,
                            'pe': pe_remaining,
                            'q_true': q_host_true_eval,
                            'q_pred': q_host_pred_eval,
                            't_from_0': t_gps_interp_eval - t_gps_interp_eval[0],
                            #'t_stamps_updates': t_stamps_updates,
                            'data_full_att_h': data_full_att_h
                        }

            return result_for_selected

        # ---- Generate datasets ----
        # Always produce the original dense 5 ms CSVs (preserves previous behavior)
        selected_result = process_quaternion_prediction(apply_sign_swap=False, do_respect_rate=False)

        # If user asked to respect update rate, also produce the per-update-rate CSVs (suffix _respectrate)
        if respect_update_rate:
            # produce the "respect rate" versions in addition to original
            process_quaternion_prediction(apply_sign_swap=False, do_respect_rate=True)

        # handle sign-swap variants the same way
        if self.sign_swap_flag:
            selected_swapped_result = process_quaternion_prediction(apply_sign_swap=True, do_respect_rate=False)
            if respect_update_rate:
                process_quaternion_prediction(apply_sign_swap=True, do_respect_rate=True)

        # ---- Visualization: keep original behavior unchanged ----
        if selected_result is not None:
            self.pe_data = selected_result
            if self.sign_swap_flag and selected_swapped_result is not None:
                self.pe_data_swapped = selected_swapped_result
            self.update_pe_visualization()
            pe_tab_index = self.graphics_tabs.indexOf(self.pe_graphics)
            if pe_tab_index != -1:
                self.graphics_tabs.setCurrentIndex(pe_tab_index)

        # ---- Vector rotation + PE plotting ----
        unit_vector = np.array([1, 0, 0])
        time_pe = self.pe_data['time']
        q_true = self.pe_data['q_true']
        q_pred = self.pe_data['q_pred']
        rotated_vectors_true = np.array([rot.rotate_with_quat(unit_vector, q) for q in q_true])
        rotated_vectors_pred = np.array([rot.rotate_with_quat(unit_vector, q) for q in q_pred])

        dot_products = np.einsum('ij,ij->i', rotated_vectors_true, rotated_vectors_pred)
        norm_true = np.linalg.norm(rotated_vectors_true, axis=1)
        norm_pred = np.linalg.norm(rotated_vectors_pred, axis=1)
        zero_mask = (norm_true == 0) | (norm_pred == 0)
        norm_true_safe = norm_true.copy(); norm_pred_safe = norm_pred.copy()
        norm_true_safe[zero_mask] = 1.0; norm_pred_safe[zero_mask] = 1.0
        cos_angles = np.clip(dot_products / (norm_true_safe * norm_pred_safe), -1.0, 1.0)
        angle = 1e6 * np.arccos(cos_angles)
        angle[zero_mask] = 0.0
        plt.plot(time_pe, angle)
        plt.xlabel('time')
        plt.ylabel('PE [µrad]')
        plt.grid()
        plt.show()

        # ---- Animation (unchanged) ----
        RPY = np.array([roll, pitch, yaw])
        q0 = att_conv.convert_ea2quat(RPY)
        if isinstance(q_true, np.ndarray) and q_true.ndim == 2 and q_true.shape[1] >= 4:
            qseq = q_true[:, :4]
        else:
            qseq = np.atleast_2d(q_true)[:, :4]
        q_full = np.vstack([q0, qseq])

        verts = np.array([[-1, -1, -1],
                        [ 1, -1, -1],
                        [ 1,  1, -1],
                        [-1,  1, -1],
                        [-1, -1,  1],
                        [ 1, -1,  1],
                        [ 1,  1,  1],
                        [-1,  1,  1]], dtype=float)
        faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])
        rng = 1.6
        ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng); ax.set_zlim(-rng, rng)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        poly = Poly3DCollection([], facecolors='skyblue', edgecolors='k', alpha=0.9)
        ax.add_collection3d(poly)
        ax.view_init(elev=20, azim=30)

        axis_colors = ['r', 'g', 'b']
        axis_lines = []
        axis_length = 1.2
        for c in axis_colors:
            ln, = ax.plot([0, 0], [0, 0], [0, 0], color=c, linewidth=2)
            axis_lines.append(ln)

        face_lines = []
        normal_length = 0.4
        for _ in faces:
            ln, = ax.plot([0, 0], [0, 0], [0, 0], color='orange', linewidth=1)
            face_lines.append(ln)

        def update(frame):
            q = q_full[frame]
            norm = np.linalg.norm(q)
            R = np.eye(3) if norm == 0 else att_conv.convert_quat2dcm(q / norm)
            rotated = verts @ R.T
            face_verts = [[rotated[idx] for idx in face] for face in faces]
            poly.set_verts(face_verts)

            body_axes = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
            tips = (body_axes @ R.T) * axis_length
            for i, ln in enumerate(axis_lines):
                x = [0.0, tips[i, 0]]
                y = [0.0, tips[i, 1]]
                z = [0.0, tips[i, 2]]
                ln.set_data(x, y)
                ln.set_3d_properties(z)

            for idx, face in enumerate(faces):
                face_pts = rotated[face]
                center = np.mean(face_pts, axis=0)
                v0 = face_pts[1] - face_pts[0]
                v1 = face_pts[2] - face_pts[0]
                n = np.cross(v0, v1)
                n_norm = np.linalg.norm(n)
                if n_norm > 0: n /= n_norm
                tip = center + n * normal_length
                ln = face_lines[idx]
                ln.set_data([center[0], tip[0]], [center[1], tip[1]])
                ln.set_3d_properties([center[2], tip[2]])

            return tuple([poly] + axis_lines + face_lines)

        try:
            speed_mult = int(self.update_rate.currentData() or 1)
        except Exception:
            speed_mult = 1
        base_interval = 5
        interval_ms = max(1, int(base_interval / float(speed_mult)))
        ani = FuncAnimation(fig, update, frames=len(q_full), interval=interval_ms, blit=False, repeat=True)
        self._pe_anim = ani
        try:
            plt.show(block=False)
        except TypeError:
            plt.show()
        return ani
    
    def quat_to_matrix(self, q):
        return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    def make_transform(self, Rmat, translation):
        T = np.eye(4)
        T[:3,:3] = Rmat
        T[:3,3] = translation
        return T
    
    def calculate_pe_new(self, active_interpolator, respect_update_rate: bool = False):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.animation import FuncAnimation
        import quaternion_slerp_squad as quat_squad

        
        """Calculate pointing error for all update_rate/latency combos, plot only selected, save all CSVs.
        Default behavior (respect_update_rate=False) preserves original dense 5 ms outputs.
        If respect_update_rate=True, the function also generates per-update-rate evaluation CSVs
        with filenames suffixed by '_respectrate.csv'.
        """
        print("*"*80)
        print("Calculate pointing error for all update_rate/latency")
        print("*"*80)
        active_interpolator  = self.active_interpolator
        print(f"Interpolation method:{active_interpolator}")
        if not self.attitude_data:
            return
        
        if self.rsepect_rate_flag:
            respect_update_rate = True

        selected_update_rate = int(self.update_rate.currentData())
        selected_latency = int(self.latency.currentData())
        update_rates = [1, 2, 5, 10]
        latencies = [0, 1, 2, 3, 4]
        selected_result = None
        selected_swapped_result = None

        # Get attitude profile info for filename
        settings_name = self.settings_combo.currentText()
        roll = float(self.roll.value())
        pitch = float(self.pitch.value())
        yaw = float(self.yaw.value())

        dt_req = 5e-3  # 5 ms (original dense sampling)
        n_digits = 3
        propagators_enabled = 1

        # True Data as generatd by astraa
        q_true      = self.attitude_data['quaternions']
        qdot_true   = self.attitude_data['quaternion_rates']

        # Common preprocessing (do once)
        t_vec = np.round(self.attitude_data['time'], n_digits)
        t_req = np.round(np.arange(t_vec[0], t_vec[-1] + dt_req, dt_req), n_digits)
        t_gps_interp = CubicSpline(t_vec, t_vec, axis=0)

            

        def process_quaternion_prediction(apply_sign_swap=False, do_respect_rate=False):
            """
            Internal worker that produces one set of CSVs for either:
                do_respect_rate == False -> dense 5 ms sampling
                do_respect_rate == True  -> sampling at 1/update_rate
            Returns result_for_selected (for plotting) if the processed combo matches selected UI combo.
            """
            result_for_selected = None
            if self.rsepect_rate_flag:
                do_respect_rate = True

            for update_rate in update_rates:
                for latency in latencies:
                    # decide timestep for evaluation
                    dt_eval = (1.0 / update_rate) if do_respect_rate else dt_req
                    t_eval = np.round(np.arange(t_vec[0], t_vec[-1] + dt_eval, dt_eval), n_digits)

                    # timing + latency
                    dt_gap_att_h = np.round(1 / update_rate, 3)
                    dt_latency = dt_gap_att_h * latency
                    t_update_arrival = np.round(np.arange(t_vec[0], t_vec[-1] + dt_gap_att_h, dt_gap_att_h), 3)

                    # file & folder
                    output_dir = os.path.join(self.outputdir, 'tables', f'{settings_name}_quatpred')
                    os.makedirs(output_dir, exist_ok=True)

                    suffix = '_respectrate' if do_respect_rate else ''
                    filename_prefix = 'swapped_true_quat' if apply_sign_swap else 'true_quat'
                    output_file = os.path.join(output_dir,
                        f'{filename_prefix}{settings_name}_roll{roll}_pitch{pitch}_yaw{yaw}_{update_rate}Hz_{latency}s_{active_interpolator}{suffix}.csv'
                    )

                    if os.path.exists(output_file):
                        print(f'File exists: {output_file}')
                        if update_rate == selected_update_rate and latency == selected_latency:
                            result_for_selected = self._load_pe_csv(output_file)
                        continue

                    t_start     = t_vec[0] + latency
                    t_end       = t_vec[-1]
                    # dt_output   = 1.0 / selected_update_rate
                    dt_output   = 1.0 / update_rate

                    # Build regular timestamp grid
                    t_key = np.arange(t_start, t_end + dt_output/2, dt_output)   
                    # Map each desired time to nearest true sample (left-side for causality)
                    indices = np.searchsorted(t_vec, t_key, side='right') - 1
                    indices = np.clip(indices, 0, len(t_vec)-1)

                    # Final delayed + downsampled data
                    t_key       = t_vec[indices]
                    q_key       = q_true[indices]
                    qdot_key    = qdot_true[indices]
                    fix_sign_swap = self.sign_swap_flag

                    if do_respect_rate:
                        q_host_true_eval = q_true[indices] # q_host_interp(t_eval)
                    else:
                        q_host_true_eval = q_true # q_host_interp(t_eval)                                          
                    q_host_pred_eval = np.zeros_like(q_host_true_eval)
                    t_stamps_updates = t_update_arrival - dt_latency                    
                   
                    # Evaluate ground truth / predicted times at t_eval
                    if active_interpolator == 'SLERP':
                        q_host_interp   = quat_slerp.make_slerp_interpolator(t_key, q_key, qdot_key, fix_sign_swap)
                        t_gps_interp_eval = t_gps_interp(t_eval)
                        q_host_pred_eval = q_host_interp(t_eval)
                        data_full_att_h = q_host_interp(t_stamps_updates)
                        # pe_remaining = att_pred.vector_angular_error(q_host_true_eval, q_host_pred_eval, v_body=np.array([0, 0, 1]))

                        pe_remaining = np.array([att_pred.vector_angular_error(q_t, q_p, np.array([0,0,1]))
                                                for q_t, q_p in zip(q_host_true_eval, q_host_pred_eval)
                                            ])
                        # pe_remaining = quat_slerp.quat_angle_error(q_host_true_eval, q_host_pred_eval)   # µrad
                        quat_error = np.array([MEKFComparator.quaternion_error(q_t, q_p)
                                               for q_t, q_p in zip(q_host_true_eval, q_host_pred_eval)])* 1e6
                    
                    elif active_interpolator == 'MOD-CUBIC-SPLINE':    
                        q_host_interp   = quat_slerp.make_cubic_spline_interpolator(t_key, q_key[:,:4], fix_sign_swap)
                        t_gps_interp_eval = t_gps_interp(t_eval)
                        q_host_pred_eval = q_host_interp(t_eval)
                        data_full_att_h = q_host_interp(t_stamps_updates)
                        # pe_remaining = att_pred.vector_angular_error(q_host_true_eval, q_host_pred_eval, v_body=np.array([0, 0, 1]))

                        pe_remaining = np.array([att_pred.vector_angular_error(q_t, q_p, np.array([0,0,1]))
                                                for q_t, q_p in zip(q_host_true_eval, q_host_pred_eval)
                                            ])
                        # pe_remaining = quat_slerp.quat_angle_error(q_host_true_eval, q_host_pred_eval)   # µrad
                        quat_error = np.array([MEKFComparator.quaternion_error(q_t, q_p)
                                               for q_t, q_p in zip(q_host_true_eval, q_host_pred_eval)])* 1e6

                    elif active_interpolator == 'CUBIC-SPLINE':               
                        q_host_interp   = CubicSpline(t_key, q_key, axis = 0)
                        t_gps_interp_eval = t_gps_interp(t_eval)
                        q_host_true_eval = q_host_interp(t_eval)

                        # t_gps_pred_eval = np.zeros_like(t_gps_interp_eval)
                        

                        # optional quaternion sign swap (keeps original logic)                        
                        if apply_sign_swap:
                            index = np.where(t_gps_interp_eval == 10)[0]
                            if len(index) > 0:
                                q_host_true_eval[index[0]:] = -q_host_true_eval[index[0]:]

                        # data used for held/propagated attitude
                        data_full_att_h = q_host_interp(t_stamps_updates)
                        
                        ii_next_att_h = 0
                        quat_interp = interp.we_interpolating()

                        # iterate through evaluation times and compute predicted quat
                        for ii, t_ii in enumerate(t_gps_interp_eval):
                            # advance held index if this eval time reached the next update arrival
                            if ii_next_att_h < len(t_update_arrival) and t_ii >= t_update_arrival[ii_next_att_h] and t_ii < t_update_arrival[-1]:
                                data_att_h = data_full_att_h[ii_next_att_h]
                                data_att_h_held = data_att_h  # for propagator off
                                if ii_next_att_h >= 1:
                                    try:
                                        quat_interp.get_quad_interpolant(
                                            t_both=t_stamps_updates[ii_next_att_h-1:ii_next_att_h+1],
                                            r_both=data_full_att_h[ii_next_att_h-1:ii_next_att_h+1, :4],
                                            v_both=(data_full_att_h[ii_next_att_h-1:ii_next_att_h+1, 4:]
                                                    if data_full_att_h.shape[1] > 4 else np.zeros((2, 4))),
                                        )
                                    except Exception as e:
                                        print(f"Interpolation error: {e}")
                                ii_next_att_h += 1

                            if ii_next_att_h >= 2:
                                if propagators_enabled:
                                    data_att_interp = quat_interp.interpolate_flexible(t_ii)
                                else:
                                    data_att_interp = data_att_h_held[:4]
                            else:
                                data_att_interp = [0, 0, 0, 0]

                            q_host_pred_eval[ii, :4] = data_att_interp
                            #t_gps_pred_eval[ii] = t_ii

                        # compute pointing error using existing att_pred helper
                        pe_remaining = [
                            att_pred.eval_pred_error(q_pred_ii, q_true_ii)[1]
                            for q_pred_ii, q_true_ii in zip(q_host_true_eval[:, :4], q_host_pred_eval[:, :4])
                        ]
                        pe_remaining = np.array(pe_remaining)
                        quat_error = np.array([MEKFComparator.quaternion_error(q_t, q_p)
                                               for q_t, q_p in zip(q_host_true_eval, q_host_pred_eval)])* 1e6

                    # save dataframe
                    # note: maintain same columns as original code
                    df_dict = {
                        'time': t_gps_interp_eval,
                        'pe': pe_remaining,
                        'quat_error': quat_error,
                        'q_true_w': q_host_true_eval[:, 0],
                        'q_true_x': q_host_true_eval[:, 1],
                        'q_true_y': q_host_true_eval[:, 2],
                        'q_true_z': q_host_true_eval[:, 3],
                        'q_true_w_dot': q_host_true_eval[:, 0],
                        'q_true_x_dot': q_host_true_eval[:, 1],
                        'q_true_y_dot': q_host_true_eval[:, 2],
                        'q_true_z_dot': q_host_true_eval[:, 3],
                        'q_pred_w': q_host_pred_eval[:, 0],
                        'q_pred_x': q_host_pred_eval[:, 1],
                        'q_pred_y': q_host_pred_eval[:, 2],
                        'q_pred_z': q_host_pred_eval[:, 3],
                    }
                    pd.DataFrame(df_dict).to_csv(output_file, index=False)
                    swap_msg = 'swapped quaternions ' if apply_sign_swap else ''
                    mode_msg = ' (respect update rate)' if do_respect_rate else ''
                    print(f'Saved {swap_msg}{output_file}{mode_msg}')

                    # if this is the UI-selected combo, capture for visualization
                    if update_rate == selected_update_rate and latency == selected_latency:
                        result_for_selected = {
                            'time': t_gps_interp_eval,
                            # 'time_key': t_key,
                            # 't_eval' :t_eval,
                            # 'q_key': q_key,
                            'pe': pe_remaining,
                            'quat_error': quat_error,
                            'q_true': q_host_true_eval,
                            'q_pred': q_host_pred_eval,
                            't_from_0': t_gps_interp_eval - t_gps_interp_eval[0],
                            # 't_stamps_updates': t_stamps_updates,
                            'data_full_att_h': data_full_att_h
                        }

            # # ====================== STATISTICS ======================
            # print("\n" + "="*80)
            # print("      REAL QUATERNION INTERPOLATION COMPARISON (vs True High-Rate Data)")
            # print("="*80)
            # print(f"Keyframes: {len(result_for_selected['time_key'])} @ ~{1/np.mean(np.diff(result_for_selected['time_key'])):.1f} Hz")
            # print(f"Truth:     {len(self.attitude_data['time'])} points @ ~{1/np.mean(np.diff(self.attitude_data['time'])):.1f} Hz")
            # print("Error metrics (µrad):")
            # print("-"*80)
            # print(f"{'Method':<12} {'Mean (µrad)':>12} {'RMS (µrad)':>12} {'Max (µrad)':>12} {'99th %ile':>12}")
            # print("-"*80)
            # #for name, err in [('SLERP', err_slerp), ('Hermite+ω', err_hermite), ('CubicSpline', err_cubicspline)]:
            # print(f"{active_interpolator:<12} {np.mean(result_for_selected['pe']):12.2f} {np.sqrt(np.mean(result_for_selected['pe']**2)):12.2f} {np.max(result_for_selected['pe']):12.1f} {np.percentile(result_for_selected['pe'], 99):12.1f}")
            # print("="*80)

            return result_for_selected

        # ---- Generate datasets ----
        # Always produce the original dense 5 ms CSVs (preserves previous behavior)
        selected_result = process_quaternion_prediction(apply_sign_swap=False, do_respect_rate=False)

        # If user asked to respect update rate, also produce the per-update-rate CSVs (suffix _respectrate)
        if respect_update_rate:
            # produce the "respect rate" versions in addition to original
            process_quaternion_prediction(apply_sign_swap=False, do_respect_rate=True)

        # handle sign-swap variants the same way
        if self.sign_swap_flag:
            selected_swapped_result = process_quaternion_prediction(apply_sign_swap=True, do_respect_rate=False)
            if respect_update_rate:
                process_quaternion_prediction(apply_sign_swap=True, do_respect_rate=True)

        # ---- Visualization: keep original behavior unchanged ----
        if selected_result is not None:
            self.pe_data = selected_result
            if self.sign_swap_flag and selected_swapped_result is not None:
                self.pe_data_swapped = selected_swapped_result
            self.update_pe_visualization()
            pe_tab_index = self.graphics_tabs.indexOf(self.pe_graphics)
            if pe_tab_index != -1:
                self.graphics_tabs.setCurrentIndex(pe_tab_index)
        ##---------------------------------------------------------
        fig, axes = plt.subplots(6, 1, figsize=(8, 6), sharex=True)
        components = ['w', 'x', 'y', 'z']
        colors = {'true': 'black', f'{active_interpolator}': 'tab:blue'}
        ['q_pred_w','q_pred_x', 'q_pred_y', 'q_pred_z'  ]
        for i in range(4):
            ax = axes[i]
            ax.plot(self.attitude_data['time'], q_true[:, i], '.', color=colors['true'], markersize=3, alpha=0.7, label='True (high-rate)')
            #ax.plot(selected_result['time_key'],  selected_result['q_key'][:, i], 'o', color='red', markersize=6, label='Keyframes' if i==0 else None)
            ax.plot(selected_result['time'], selected_result['q_pred'][:, i], '-', color=colors[f'{active_interpolator}'], linewidth=1.5, label=f'{active_interpolator}')
            ax.set_ylabel(f'q[{components[i]}]')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=9)

        # Norm plot
        axes[4].plot(self.attitude_data['time'], np.linalg.norm( q_true, axis=1), '.', color='black', alpha=0.7, label='True')
        axes[4].plot(selected_result['time'], np.linalg.norm(selected_result['q_pred'], axis=1), color=colors[f'{active_interpolator}'], label=f'{active_interpolator}')
        axes[4].axhline(1.0, color='red', linestyle='--', alpha=0.6)
        axes[4].set_ylabel('||q||')
        axes[4].legend(fontsize=9)
        axes[4].grid(True, alpha=0.3)

        # Error plot
        axes[5].plot(selected_result['time'], selected_result['pe'], color=colors[f'{active_interpolator}'], linewidth=2, label=f"{active_interpolator}  | Max: {(self.pe_data['pe']).max():.1f} µrad")
        #axes[5].plot(t_eval, err_hermite, color=colors['hermite'], linewidth=2.5, label=f'Hermite | Max: {err_hermite.max():.1f} µrad')
        #axes[5].plot(t_eval, err_cubicspline, '--', color='tab:green', linewidth=1.5, label=f'Cubic Spline | Max: {err_cubicspline.max():.1f} µrad')
        axes[5].set_ylabel('Angular Error [µrad]')
        axes[5].set_xlabel('Time [s]')
        axes[5].set_yscale('log')
        axes[5].grid(True, alpha=0.3)
        axes[5].legend(fontsize=10)
        plt.suptitle('Quaternion Interpolation:\n'
             'Evaluated against original high-rate ground truth (no fake interpolation!)', fontsize=10, y=0.98)
        plt.tight_layout()
        output_dir = os.path.join(self.outputdir, 'tables', f'{settings_name}_quatpred')
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(os.path.join(output_dir, f'{settings_name}_quaternion_interpolation_error.png'), dpi=200, bbox_inches='tight')
        plt.show()
        ##---------------------------------------------------------

       


        # ---- Animation (unchanged) ----
        # RPY = np.array([roll, pitch, yaw])
        # q0 = att_conv.convert_ea2quat(RPY)
        # if isinstance(q_true, np.ndarray) and q_true.ndim == 2 and q_true.shape[1] >= 4:
        #     qseq = q_true[:, :4]
        # else:
        #     qseq = np.atleast_2d(q_true)[:, :4]
        # q_full = np.vstack([q0, qseq])

        # verts = np.array([[-1, -1, -1],
        #                 [ 1, -1, -1],
        #                 [ 1,  1, -1],
        #                 [-1,  1, -1],
        #                 [-1, -1,  1],
        #                 [ 1, -1,  1],
        #                 [ 1,  1,  1],
        #                 [-1,  1,  1]], dtype=float)
        # faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]

        # fig = plt.figure(figsize=(6,6))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_box_aspect([1,1,1])
        # rng = 1.6
        # ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng); ax.set_zlim(-rng, rng)
        # ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        # poly = Poly3DCollection([], facecolors='skyblue', edgecolors='k', alpha=0.9)
        # ax.add_collection3d(poly)
        # ax.view_init(elev=20, azim=30)

        # axis_colors = ['r', 'g', 'b']
        # axis_lines = []
        # axis_length = 1.2
        # for c in axis_colors:
        #     ln, = ax.plot([0, 0], [0, 0], [0, 0], color=c, linewidth=2)
        #     axis_lines.append(ln)

        # face_lines = []
        # normal_length = 0.4
        # for _ in faces:
        #     ln, = ax.plot([0, 0], [0, 0], [0, 0], color='orange', linewidth=1)
        #     face_lines.append(ln)

        # def update(frame):
        #     q = q_full[frame]
        #     norm = np.linalg.norm(q)
        #     R = np.eye(3) if norm == 0 else att_conv.convert_quat2dcm(q / norm)
        #     rotated = verts @ R.T
        #     face_verts = [[rotated[idx] for idx in face] for face in faces]
        #     poly.set_verts(face_verts)

        #     body_axes = np.array([[1.0, 0.0, 0.0],
        #                         [0.0, 1.0, 0.0],
        #                         [0.0, 0.0, 1.0]])
        #     tips = (body_axes @ R.T) * axis_length
        #     for i, ln in enumerate(axis_lines):
        #         x = [0.0, tips[i, 0]]
        #         y = [0.0, tips[i, 1]]
        #         z = [0.0, tips[i, 2]]
        #         ln.set_data(x, y)
        #         ln.set_3d_properties(z)

        #     for idx, face in enumerate(faces):
        #         face_pts = rotated[face]
        #         center = np.mean(face_pts, axis=0)
        #         v0 = face_pts[1] - face_pts[0]
        #         v1 = face_pts[2] - face_pts[0]
        #         n = np.cross(v0, v1)
        #         n_norm = np.linalg.norm(n)
        #         if n_norm > 0: n /= n_norm
        #         tip = center + n * normal_length
        #         ln = face_lines[idx]
        #         ln.set_data([center[0], tip[0]], [center[1], tip[1]])
        #         ln.set_3d_properties([center[2], tip[2]])

        #     return tuple([poly] + axis_lines + face_lines)

        # try:
        #     speed_mult = int(self.update_rate.currentData() or 1)
        # except Exception:
        #     speed_mult = 1
        # base_interval = 5
        # interval_ms = max(1, int(base_interval / float(speed_mult)))
        # ani = FuncAnimation(fig, update, frames=len(q_full), interval=interval_ms, blit=False, repeat=True)
        # self._pe_anim = ani
        # try:
        #     plt.show(block=False)
        # except TypeError:
        #     plt.show()
        # return ani
        #-------------------------------------------------------------------------------------------------------------
        # ---- Animation (fixed) ----
        # ------------------------------------------------------------
        # 2️⃣  BUILD q_full (initial + true sequence) -----------------
        # ------------------------------------------------------------
        # RPY = np.array([roll, pitch, yaw])        
        # q0 = R.from_euler('xyz', RPY, degrees=True).as_quat()[[3, 0, 1, 2]]   # (w, x, y, z)
        # q0 = q0.reshape(1, 4)                                   # (1,4)

        # qt = np.asarray(q_true, dtype=float)
        # if qt.ndim == 1:
        #     qt = qt.reshape(1, 4)
        # elif qt.ndim == 2 and qt.shape[1] >= 4:
        #     qt = qt[:, :4]
        # else:
        #     qt = np.atleast_2d(qt)[:, :4]

        # norms = np.linalg.norm(qt, axis=1, keepdims=True)
        # qt = np.where(norms == 0, qt, qt / norms)                # normalise

        # q_full = np.vstack([q0, qt])                            # (N+1,4)

        # # ------------------------------------------------------------
        # # 3️⃣  CUBE GEOMETRY ------------------------------------------------
        # # ------------------------------------------------------------
        # verts = np.array([
        #     [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        #     [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
        # ], dtype=float)

        # faces = [
        #     [0,1,2,3], [4,5,6,7],
        #     [0,1,5,4], [2,3,7,6],
        #     [1,2,6,5], [0,3,7,4]
        # ]
        # # ------------------------------------------------------------
        # # 4️⃣  FIGURE / AXES ------------------------------------------------
        # # ------------------------------------------------------------
        # fig = plt.figure(figsize=(6,6))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_box_aspect([1,1,1])
        # rng = 1.6
        # ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng); ax.set_zlim(-rng, rng)
        # ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        # poly = Poly3DCollection([], facecolors='skyblue', edgecolors='k', alpha=0.9)
        # ax.add_collection3d(poly)

        # # optional axis / normal helpers
        # axis_colors = ['r','g','b']
        # axis_len = 1.2
        # axis_lines = [ax.plot([0,0],[0,0],[0,0],c=c,lw=2)[0] for c in axis_colors]

        # normal_len = 0.4
        # face_lines = [ax.plot([0,0],[0,0],[0,0],c='orange',lw=1)[0] for _ in faces]

        # # ------------------------------------------------------------
        # # 5️⃣  UPDATE CALLBACK ---------------------------------------------
        # # ------------------------------------------------------------
        # def update(frame_idx):
        #     q = q_full[frame_idx]               # (4,) unit quaternion
        #     Rmat = R.from_quat(q).as_matrix()   # 3×3 rotation matrix

        #     rotated = verts @ Rmat.T
        #     face_verts = [[rotated[i] for i in f] for f in faces]
        #     poly.set_verts(face_verts)

        #     # body axes
        #     body_axes = np.eye(3)
        #     tips = (body_axes @ Rmat.T) * axis_len
        #     for i, ln in enumerate(axis_lines):
        #         ln.set_data([0.0, tips[i,0]], [0.0, tips[i,1]])
        #         ln.set_3d_properties([0.0, tips[i,2]])

        #     # face normals
        #     for idx, f in enumerate(faces):
        #         pts = rotated[f]
        #         centre = pts.mean(axis=0)
        #         n = np.cross(pts[1]-pts[0], pts[2]-pts[0])
        #         n_norm = np.linalg.norm(n)
        #         if n_norm > 1e-9:
        #             n = n / n_norm
        #         tip = centre + n * normal_len
        #         face_lines[idx].set_data([centre[0], tip[0]],
        #                                 [centre[1], tip[1]])
        #         face_lines[idx].set_3d_properties([centre[2], tip[2]])

        #     return (poly, ) + tuple(axis_lines) + tuple(face_lines)

        # ------------------------------------------------------------
        # 6️⃣  ANIMATION ----------------------------------------------------
        # ------------------------------------------------------------
        # interval_ms = 50                     # ~20 fps
        # anim = FuncAnimation(fig, update,
        #                     frames=len(q_full),
        #                     interval=interval_ms,
        #                     blit=False,
        #                     repeat=True)

        # # keep a reference so it isn’t garbage‑collected
        # plt.show()
        #-------------------------------------------------------------------------------------------------------------


        # q_host_interp = CubicSpline(t_vec, q_true, axis=0)
        # q_host_interp = quat_squad.make_quaternion_integrator(t_vec, self.attitude_data['quaternions'], self.attitude_data['quaternion_rates'])

        #self.active_interpolator = self.quaternion_integrator.currentText() if self.quaternion_integrator.currentText() != "CubicSpline" else "CubicSpline"
        # q_host_interp = CubicSpline(t_vec, q_true, axis=0) if self.active_interpolator == "CubicSpline" else quat_squad.make_quaternion_integrator(t_vec, self.attitude_data['quaternions'], self.attitude_data['quaternion_rates'])
        #print(f"Active quaternion integrator: {self.active_interpolator}")
        #q_host_interp = CubicSpline(t_vec, q_true, axis=0) if self.active_interpolator == "CubicSpline" else quat_squad.make_quaternion_integrator(t_vec, self.attitude_data['quaternions'], self.attitude_data['quaternion_rates'])  

    def calculate_pe_ephemeris(self):#,host,target):
        from collections import deque
        from matplotlib.animation import FuncAnimation        

        def propagate_with_updates(t_init,
                            prop_duration,
                            prop_timeout,
                            t_gps,
                            r_target,
                            v_target,
                            odeRK,
                            orbit_prop,
                            step=1,
                            update_times=None):

            updates = deque(update_times if update_times is not None else [])

            prop_trajectory = []
            true_trajectory = []
            t_all = []

            r_prop_1 = None
            v_prop_1 = None

            iteration = 0

            while t_init < prop_timeout:

                next_update = updates[0] if updates else None

                if next_update is not None and next_update <= t_init + prop_duration:
                    prop_end = next_update
                    update = True
                else:
                    prop_end = min(t_init + prop_duration, prop_timeout)
                    update = False

                indices = np.arange(t_init, prop_end + 1, step)
                if len(indices) == 0:
                    print("⚠️ Empty tspan, stopping")
                    break

                tspan = t_gps[indices]
                true_slice = r_target[indices]

                if iteration == 0 or update:
                    init_idx = t_init
                    v_sv_op = np.hstack([r_target[init_idx], v_target[init_idx]])
                else:
                    v_sv_op = np.hstack([r_prop_1[-1], v_prop_1[-1]])

                t, path = self.odeRK(orbit_prop, tspan, v_sv_op, substeps=30)

                r_prop_1 = path[:, :3]
                v_prop_1 = path[:, 3:6]

                print(f"Propagated indices {t_init}→{prop_end} | time {tspan[0]:.1f}→{tspan[-1]:.1f} ({tspan[-1]-tspan[0]:.1f}s) | steps={len(tspan)} | update={update}")

                prop_slice = r_prop_1

                assert prop_slice.shape[0] == true_slice.shape[0]

                prop_trajectory.append(prop_slice)
                true_trajectory.append(true_slice)
                t_all.append(t)

                # === FIXED ADVANCE LOGIC ===
                if update:
                    t_init = next_update
                    updates.popleft()
                else:
                    # After normal propagation, next start = last index we just propagated + 1
                    t_init = indices[-1]# prop_end + 1

                iteration += 1

            prop_trajectory = np.vstack(prop_trajectory) if prop_trajectory else np.array([])
            true_trajectory = np.vstack(true_trajectory) if true_trajectory else np.array([])
            t_all = np.hstack(t_all) if t_all else np.array([])

            return t_all, prop_trajectory, true_trajectory
        
        def debug_timeline(t_init, prop_duration, updates, timeout):
   
            updates = deque(updates)
            timeline = []

            while t_init < timeout:
                next_update = updates[0] if updates else None

                if next_update is not None and next_update <= t_init + prop_duration:
                    prop_end = next_update - 1
                    timeline.append((t_init, prop_end, "UPDATE"))
                    t_init = next_update
                    updates.popleft()
                else:
                    prop_end = min(t_init + prop_duration - 1, timeout)
                    timeline.append((t_init, prop_end, "NO UPDATE"))
                    t_init = prop_end + 1

            return timeline

        def plot_timeline(timeline, updates, timeout):
            fig, ax = plt.subplots(figsize=(12, 2))

            # Plot propagation segments
            for (start, end, kind) in timeline:
                color = "tab:blue" if kind == "NO UPDATE" else "tab:orange"
                ax.hlines(1, start, end, linewidth=8, color=color)

                # Label segment
                ax.text((start + end)/2, 1.05, f"{start}-{end}",
                        ha='center', va='bottom', fontsize=8)

            # Plot update lines
            for u in updates:
                ax.axvline(u, color='red', linestyle='--', alpha=0.7)
                ax.text(u, 0.85, f"U@{u}", rotation=90,
                        ha='center', va='top', fontsize=8, color='red')

            # Formatting
            ax.set_ylim(0.7, 1.3)
            ax.set_xlim(0, timeout)
            ax.set_yticks([])
            ax.set_xlabel("Time [sec]")
            ax.set_title("Propagation Timeline with Updates")

            # Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='tab:blue', lw=6, label='No Update'),
                Line2D([0], [0], color='tab:orange', lw=6, label='Pre-Update Propagation'),
                Line2D([0], [0], color='red', lw=2, linestyle='--', label='Update Arrival')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            #plt.show()

        def animate_propagation(t_all, true_traj, prop_traj, updates):
            fig, ax = plt.subplots()

            ax.set_title("Propagation vs Truth (Animated)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            true_line, = ax.plot([], [], 'g-', label="True")
            prop_line, = ax.plot([], [], 'b--', label="Propagated")
            update_scatter = ax.scatter([], [], color='red', label="Updates")

            ax.legend()

            def init():
                ax.set_xlim(np.min(true_traj[:,0]), np.max(true_traj[:,0]))
                ax.set_ylim(np.min(true_traj[:,1]), np.max(true_traj[:,1]))
                return true_line, prop_line, update_scatter

            def update(frame):
                true_line.set_data(true_traj[:frame, 0], true_traj[:frame, 1])
                prop_line.set_data(prop_traj[:frame, 0], prop_traj[:frame, 1])

                # mark update points
                update_idx = [i for i, t in enumerate(t_all[:frame]) if t in updates]
                if update_idx:
                    update_scatter.set_offsets(prop_traj[update_idx, :2])

                return true_line, prop_line, update_scatter

            ani = FuncAnimation(fig, update, frames=len(t_all),
                                init_func=init, interval=50, blit=True)

            #plt.show()
            return ani

        def plot_error(t_rel, true_traj, prop_traj, updates, link_distance_m=100000.0):                                               

            """
            Plot pointing error in micro-radians
            link_distance: distance to the target satellite in meters (default 100 km)
            """
            
            # Position difference in meters
            delta_r = (true_traj - prop_traj)*1e3  # convert km to m
            pos_error_m = np.linalg.norm(delta_r, axis=1)
            
            # Pointing error in radians → micro-radians
            pointing_error_urad = 1e6 * np.arctan2(pos_error_m, link_distance_m)
        
            
            plt.figure(figsize=(10, 5))
            plt.plot(t_rel, pointing_error_urad, label="Pointing Error", color='blue', linewidth=2)
            
            # Mark update times
            for u in updates:
                plt.axvline(u, color='red', linestyle='--', alpha=0.7, label='Update' if u == updates[0] else "")
            
            plt.title("Pointing Error vs Time (link distance = 100 km)")
            plt.xlabel("Time [sec]")
            plt.ylabel("Pointing Error [µrad]")
            # plt.grid(True)
            plt.legend()
            plt.tight_layout()

            fig,ax = plt.subplots(4,1, sharex=True, figsize=(8,6))
            ax[0].plot(t_rel, delta_r[:,0], label="True X", color='green')
            ax[1].plot(t_rel, delta_r[:,1], label="True Y", color='orange')
            ax[2].plot(t_rel, delta_r[:,2], label="True Z", color='purple')
            ax[3].plot(t_rel, pos_error_m, label="$\delta", color='black', alpha=0.5)
            for u in updates:
                for a in ax:
                    a.axvline(u, color='red', linestyle='--', alpha=0.7)
            ax[0].set_title("Position Error Components vs Time")
            ax[2].set_xlabel("Time [sec]")
            ax[0].set_ylabel("Error X [m]")
            ax[1].set_ylabel("Error Y [m]")
            ax[2].set_ylabel("Error Z [m]")
            ax[3].set_ylabel("$\delta [m]")
            ax[1].grid(True)
            ax[0].grid(True)
            ax[2].grid(True)
            ax[3].grid(True)
            plt.tight_layout()


            return pointing_error_urad
            
        def compute_update_jumps(t_all, true_traj, prop_traj, updates):
        
            error = np.linalg.norm(true_traj - prop_traj, axis=1)

            jumps = []

            for u in updates:
                idx = np.where(t_all == u)[0]
                if len(idx) == 0:
                    continue

                i = idx[0]

                if i > 0:
                    before = error[i-1]
                    after = error[i]
                    jumps.append((u, before, after, before - after))

            return jumps

        outputdir = self.run_orbit_simulation(hp_setting=1,sim_time=24*3600, pe_flag = True)
        """Calculate pointing error from ephemeris"""
        print("*"*80)
        print("Calculate pointing error from ephemeris")
        print("*"*80)
        # if not  self.simulation_data:
        #     return
        
        # get satellite hostory data from disk
        # Define the path to the .dat file
        file_path = os.path.join(self.outputdir, 'pointing_error', 'state_history.dat')#'examples/output_data/pointing_error/state_history.dat'
        load.open_dat(outputdir)
        # Load the data into a DataFrame (you can inspect the header if needed)
        sat_data = pd.read_csv(file_path, sep='\t', header = None, comment='#')
            
            
        # Get parameters
        update_step = self.ephm_update_rate.value()
        latency = self.ephm_latency.value()
        update_no = self.update_no.value()  # Number of updates

        importlib.reload(j2prop)
        # update_step = 10

        #host_chosen   = self.host_sat.currentText()
        target_chosen = self.sat_pe.currentText()
        link_distance_m = self.link_distance.value()*1e3

        data_raw, simulation_parameters = dputil.load_constellation_data(full_path = outputdir)

        t_j2000 = data_raw[:,0]
        t_gps   = t_conv.j2000_to_gps(t_j2000)  #t_j2000 + t_conv.dt_j2000tt2gps()
         
        r_target = sat_data.iloc[:,simulation_parameters['r_index'][target_chosen]].to_numpy() *1e-3                     #in km             #[data_raw[:,simulation_parameters['r_index'][target_chosen]]
        v_target = sat_data.iloc[:,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]].to_numpy() *1e-3    #in km/s      #[data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]]
        
        # Simulation parameters
        start_point = 0# 2587  # Time index for ~250km range
        print(f"Shifting HOST initial zero position to: {start_point} sec")
        print("So for simulation this will be HOST initial position.")

        # Get target delay
        target_delay = latency# int(input("\nTARGET initial ephemeris delay in sec [max. 50]? (If no delay, put 0): "))
        if target_delay > 50:
            print("Error: Target delay exceeds maximum of 50 seconds.")
            exit(1)

        # Target initial conditions
        t_init = latency # target_delay
        
        ## Propagation parameters
        #update_no = int(input("\nHow many ephemeris updates will be received? : "))
        prop_duration = self.prop_dur.value()  # sec
        prop_timeout = start_point + 100
        print(f"SDA limiting propagation timeout (considering HOST position shift by {start_point}): {prop_timeout} sec")
        # prop_end = prop_duration

        updates = t_init + np.array([update_step * (i + 1) for i in range(update_no)])

        # Convert updates to GPS time for proper alignment
        # updates_gps = t_gps[updates.astype(int)]

        t_all, prop_traj, true_traj = propagate_with_updates(
            t_init=t_init,
            prop_duration=prop_duration,
            prop_timeout=prop_timeout,
            t_gps=t_gps,
            r_target=r_target,
            v_target=v_target,
            odeRK=self.odeRK,
            orbit_prop=self.orbit_prop,
            update_times=updates
        )
        # Compute error jumps
        jumps = compute_update_jumps(t_all, true_traj, prop_traj, updates)  # use index-based updates
        for u, before, after, improvement in jumps:
            print(f"Update @ {u:.1f}: error {before:.6f} → {after:.6f} (Δ = {improvement:.6f})")

        # Plot error
        t_rel = t_all - t_all[0]
        updates_rel = np.array(updates)   # since updates are indices starting from 0

        #plot_error(t_rel, true_traj, prop_traj, updates_rel)
        pointing_error = plot_error(t_rel, true_traj, prop_traj, updates_rel, link_distance_m)
        # PE = compute_pointing_error(t_rel,true_traj, prop_traj)

        # Optional: print some statistics
        print(f"Max pointing error: {np.max(pointing_error):.2f} µrad")
        print(f"Mean pointing error: {np.mean(pointing_error):.2f} µrad")
        # print(f" Pointing error: {np.max(PE):.2f} µrad")

        plot_timeline(debug_timeline(t_init, prop_duration, updates, prop_timeout), updates, prop_timeout)
        animate_propagation(t_all, true_traj, prop_traj, updates)
        
        



        # v_sv_op = np.hstack([r_target[t_init, :], v_target[t_init, :]])

        # # Propagation parameters
        # #update_no = int(input("\nHow many ephemeris updates will be received? : "))
        # prop_duration = self.prop_dur.value()  # sec
        # prop_timeout = start_point + 100
        # print(f"SDA limiting propagation timeout (considering HOST position shift by {start_point}): {prop_timeout} sec")
        # prop_end = prop_duration

        # # Initialize arrays
        # step = 1
        # state = []

        # prop_trajectory = []
        # true_trajectory =[]
        # t1 = []
        # pos_init = start_point
        # t_init = start_point  # MATLAB: target_delay = start_point (when input is 0)
        # r_prop_1 = np.zeros((1, 3))
        # v_prop_1 = np.zeros((1, 3))
        # update_no = self.update_no.value()  # Number of updates

        # # MATLAB: new_data = t_init + [10, 20, 30]
        # #if update_no:
        # new_data = t_init + np.array([update_step * (i + 1) for i in range(update_no)])#np.array([10, 20, 30])
        # #else:
        #  #   new_data = t_init

        # i = 1        
        # print(f'{update_no} updates will arrive at:{new_data} sec.')

        ## NEW PROPAGTION
        # update_step = 5
        # update_no = 3

       
        
        # updates = t_init + np.array([update_step * (i + 1) for i in range(update_no)])
        
        # t_all, prop_traj, true_traj = propagate_with_updates(
        #     t_init=t_init,
        #     prop_duration=prop_duration,
        #     prop_timeout=prop_timeout,
        #     t_gps=t_gps,
        #     r_target=r_target,
        #     v_target=v_target,
        #     odeRK=self.odeRK,
        #     orbit_prop=self.orbit_prop,
        #     update_times=updates
        # )
        # jumps = compute_update_jumps(t_all, true_traj, prop_traj, updates)

        # for u, before, after, improvement in jumps:
        #     print(f"Update @ {u}: error {before:.3f} → {after:.3f} (Δ = {improvement:.3f})")

        # timeline = debug_timeline(
        #     t_init=0,
        #     prop_duration=10,
        #     updates=[5, 12, 25],
        #     timeout=40
        # )

        # if isinstance(true_traj, list):
        #     true_traj = np.vstack(true_traj)

        # if isinstance(prop_traj, list):
        #     prop_traj = np.vstack(prop_traj)

        # # --- make sure updates is a simple list ---
        # updates = list(updates)

        # # --- normalize time ---
        # t_rel = t_all - t_all[0]

        # # --- plots ---
        # plot_error(t_rel, true_traj, prop_traj, updates)        
        # pointing_error_arcsec = compute_pointing_error(true_traj, prop_traj)
        # plt.figure(figsize=(10,4))
        # plt.plot(t_rel, pointing_error_arcsec, label="Pointing Error (arcsec)", color='purple')
        # plt.xlabel("Time [s]")
        # plt.ylabel("Pointing Error [arcsec]")
        # plt.title("Pointing Error Over Time")
        # #plt.grid(True)
        # plt.legend()


        # --- animation ---
        #animate_propagation(t_all, true_traj, prop_traj, updates)

        # --- sanity check ---
        print("Shapes:", t_all.shape, true_traj.shape, prop_traj.shape)

        # --- CALL FUNCTIONS ---
        # plot_error(t_all, true_traj, prop_traj, updates)
        # animate_propagation(t_all, true_traj, prop_traj, updates)

        # plot_timeline(timeline, updates=[5, 12, 25], timeout=40)
        # for seg in timeline:
        #     print(seg)
        # END NEW PROPAGATION
        ## PROPAGATION LOOP- ORIGINAL
        # while prop_end <= prop_timeout:
        #     if i <= len(new_data) and new_data[i-1] <= start_point + prop_duration:
        #         prop_end = new_data[i-1]
        #         update = True
        #         print(f'\nt_new_data at: {prop_end}, so propagating up to: {prop_end-1} [sec]')
        #     else:
        #         prop_end = start_point + prop_duration - 1
        #         print(f"DEBUG: prop_end: {prop_end}, start_point: {start_point}, prop_duration: {prop_duration}")
        #         update = False
        #         print(f'\nNo new data within {prop_duration} sec, so propagating to: {prop_end} [sec]')

        #     #print(f't_init: {t_init}')
        #     # tspan = time[t_init:prop_end+1:step]
        #     tspan = t_gps[t_init:prop_end+1:step]
        #     # tspan_single = np.array(tspan, dtype=np.float32)
        #     # temp.append(t_init)

        #     if update or i < 2:
        #         v_sv_op = np.hstack([r_target[t_init], v_target[t_init]])
        #         #v_sv_op_single = np.array(v_sv_op, dtype=np.float32)
        #     else:
        #         v_sv_op = np.hstack([r_prop_1[-1, :], v_prop_1[-1, :]])
        #         #v_sv_op_single = np.array(v_sv_op, dtype=np.float32)

        #     t, path1 = self.odeRK(self.orbit_prop, tspan, v_sv_op)
        #     # t_single, path1_single = self.odeRK_single(self.orbit_prop, tspan_single, v_sv_op_single)
        #     print(f"{i}# : Propagating from {tspan[0]} to {tspan[-1]} for {tspan[-1] -tspan[0]} s")

        #     r_prop_1 = path1[:, :3]           
        #     v_prop_1 = path1[:, 3:6]
            
        #     # Ensure matching shapes for stacking
        #     true_slice    = r_target[t_init:t_init+len(t), :]
        #     # propagated Target Traj.
        #     prop_slice = r_prop_1
        #     # true_slice = target_slice

        #     prop_trajectory = np.vstack([prop_trajectory, prop_slice]) if len(prop_trajectory) >0 else prop_slice
        #     true_trajectory = np.vstack([true_trajectory, true_slice]) if len(true_trajectory) >0 else true_slice
        #     t1 = np.hstack([t1, t]) if len(t1) > 0 else t

        #     i += 1
        #     t_init = prop_end#-1
        #     prop_end = t_init + prop_duration
        #     start_point = prop_end
        #     if prop_end > prop_timeout:
        #         print(f"Prop_end time {prop_end} > timeout:{prop_timeout}")
        #         prop_end = prop_timeout
        #         print(f"Proppagating till: {prop_end}")
        #         break

        # Plot pointing error
        # fig2, ax = plt.subplots(4,1)
        # ax[0].plot((true_trajectory[:, 0] - prop_trajectory[:, 0])*1e3)
        # ax[1].plot((true_trajectory[:, 1] - prop_trajectory[:, 1])*1e3)
        # ax[2].plot((true_trajectory[:, 2] - prop_trajectory[:, 2])*1e3)
        # ax[3].plot(np.linalg.norm((true_trajectory - prop_trajectory)*1e3, axis=1), label = 'd_r')
        # ax[0].set_title('Propagation position error [m]')
        # ax[0].set_ylabel('dx [m]')
        # ax[1].set_ylabel('dy [m]')
        # ax[2].set_ylabel('dz [m]')
        # ax[3].set_ylabel('dr [m]')
        # ax[3].set_xlabel('Time [s]')
        # ax[0].grid(True)
        # ax[1].grid(True)
        # ax[2].grid(True)
        # ax[3].grid(True)

        # plt.figure()
        # plt.plot(1e6*np.arctan2(np.linalg.norm((true_trajectory - prop_trajectory)*1e3, axis=1), link_distance))
        # plt.xlabel('Time [s]')
        # plt.ylabel('PE [µrads]')
        # plt.title(f"Link distance:{link_distance*1e-3}km, Eph-updates:{update_no}, update-rate:{update_step}")
        plt.grid(True)
        plt.show() 
    
    def plot_3d_groundstation(self, ground_track=None, passes=None, selected_gs=None, satellite=None):
        import datetime
        if ground_track is None or passes is None or selected_gs is None:
            print("Error: Missing required data for 3D groundstation plot.")
            return

        # Support altitude in ground_track (Nx3: lon, lat, alt_m); fall back to 400 km LEO
        if ground_track.shape[1] >= 3:
            longitudes, latitudes, altitudes = ground_track[:, 0], ground_track[:, 1], ground_track[:, 2]
        else:
            longitudes, latitudes = ground_track.T
            altitudes = [400000.0] * len(longitudes)

        gs_lon = self.gs_data['longitude_deg']
        gs_lat = self.gs_data['latitude_deg']
        gs_alt = self.gs_data.get('altitude_m', 0)

        pass_start_lons = [longitudes[p['start_idx']] for p in passes]
        pass_start_lats = [latitudes [p['start_idx']] for p in passes]
        pass_start_alts = [altitudes [p['start_idx']] for p in passes]
        pass_end_lons   = [longitudes[p['end_idx']] for p in passes]
        pass_end_lats   = [latitudes [p['end_idx']] for p in passes]
        pass_end_alts   = [altitudes [p['end_idx']] for p in passes]

        start_time = datetime.datetime.utcnow()
        time_iso   = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Pass intervals (index pairs) for JS dynamic link lines
        # pass_intervals_js = [{'start_idx': p['start_idx'], 'end_idx': p['end_idx']} for p in passes]
        pass_intervals_js = [
            {'start_idx': int(p['start_idx']), 'end_idx': int(p['end_idx'])}
            for p in passes
        ]
        print(f"Ground Station: {selected_gs}\t Satellite: {satellite}")

        # cesium_html = f'''
        # <html>
        # <head>
        #     <script src="https://cesium.com/downloads/cesiumjs/releases/1.106/Build/Cesium/Cesium.js"></script>
        #     <link href="https://cesium.com/downloads/cesiumjs/releases/1.106/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
        # </head>
        # <body>
        #     <div id="cesiumContainer" style="width:100%; height:900px;"></div>
        #     <script>
        #         var longitudes    = [{', '.join(str(v) for v in longitudes)}];
        #         var latitudes     = [{', '.join(str(v) for v in latitudes)}];
        #         var altitudes     = [{', '.join(str(v) for v in altitudes)}];
        #         var passIntervals = {pass_intervals_js};

        #         var viewer = new Cesium.Viewer('cesiumContainer', {{
        #             timeline: true,
        #             animation: true
        #         }});
        #         viewer.scene.globe.enableLighting = true;

        #         // ── Ground track polyline at real altitude ──────────────────────────
        #         var groundTrackPositions = [];
        #         for (var i = 0; i < longitudes.length; i++) {{
        #             groundTrackPositions.push(
        #                 Cesium.Cartesian3.fromDegrees(longitudes[i], latitudes[i], altitudes[i])
        #             );
        #         }}
        #         viewer.entities.add({{
        #             name: 'Ground Track',
        #             polyline: {{
        #                 positions: groundTrackPositions,
        #                 width: 2,
        #                 material: Cesium.Color.BLACK.withAlpha(0.6),
        #                 clampToGround: false
        #             }}
        #         }});

        #         // ── Ground track projection on surface (faint) ──────────────────────
        #         var surfacePositions = [];
        #         for (var i = 0; i < longitudes.length; i++) {{
        #             surfacePositions.push(
        #                 Cesium.Cartesian3.fromDegrees(longitudes[i], latitudes[i], 0)
        #             );
        #         }}
        #         viewer.entities.add({{
        #             name: 'Ground Track Projection',
        #             polyline: {{
        #                 positions: surfacePositions,
        #                 width: 1,
        #                 material: Cesium.Color.GRAY.withAlpha(0.3),
        #                 clampToGround: true
        #             }}
        #         }});

        #         // ── Ground station marker ───────────────────────────────────────────
        #         viewer.entities.add({{
        #             name: '{selected_gs}',
        #             position: Cesium.Cartesian3.fromDegrees({gs_lon}, {gs_lat}, {gs_alt}),
        #             point: {{ pixelSize: 15, color: Cesium.Color.RED }},
        #             label: {{
        #                 text: '{selected_gs}',
        #                 font: '16px sans-serif',
        #                 fillColor: Cesium.Color.RED,
        #                 style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        #                 outlineWidth: 2,
        #                 verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        #                 pixelOffset: new Cesium.Cartesian2(0, -20)
        #             }}
        #         }});

        #         // ── Pass start markers (green) at real altitude ─────────────────────
        #         var passStartPositions = [
        #             {', '.join(f'Cesium.Cartesian3.fromDegrees({lon}, {lat}, {alt})' for lon, lat, alt in zip(pass_start_lons, pass_start_lats, pass_start_alts))}
        #         ];
        #         for (var i = 0; i < passStartPositions.length; i++) {{
        #             viewer.entities.add({{
        #                 name: 'Downlink Start',
        #                 position: passStartPositions[i],
        #                 point: {{ pixelSize: 12, color: Cesium.Color.GREEN }},
        #                 label: {{
        #                     text: 'AOS',
        #                     font: '14px sans-serif',
        #                     fillColor: Cesium.Color.GREEN,
        #                     style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        #                     outlineWidth: 2,
        #                     verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        #                     pixelOffset: new Cesium.Cartesian2(0, -10)
        #                 }}
        #             }});
        #         }}

        #         // ── Pass end markers (orange) at real altitude ──────────────────────
        #         var passEndPositions = [
        #             {', '.join(f'Cesium.Cartesian3.fromDegrees({lon}, {lat}, {alt})' for lon, lat, alt in zip(pass_end_lons, pass_end_lats, pass_end_alts))}
        #         ];
        #         for (var i = 0; i < passEndPositions.length; i++) {{
        #             viewer.entities.add({{
        #                 name: 'Downlink End',
        #                 position: passEndPositions[i],
        #                 point: {{ pixelSize: 12, color: Cesium.Color.ORANGE }},
        #                 label: {{
        #                     text: 'LOS',
        #                     font: '14px sans-serif',
        #                     fillColor: Cesium.Color.ORANGE,
        #                     style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        #                     outlineWidth: 2,
        #                     verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        #                     pixelOffset: new Cesium.Cartesian2(0, -10)
        #                 }}
        #             }});
        #         }}

        #         // ── Animated satellite with real altitude ───────────────────────────
        #         var clockStart = Cesium.JulianDate.fromIso8601('{time_iso}');
        #         var clockStop  = Cesium.JulianDate.addSeconds(clockStart, longitudes.length - 1, new Cesium.JulianDate());
        #         viewer.clock.startTime   = clockStart.clone();
        #         viewer.clock.stopTime    = clockStop.clone();
        #         viewer.clock.currentTime = clockStart.clone();
        #         viewer.clock.clockRange  = Cesium.ClockRange.CLAMPED;
        #         viewer.clock.multiplier  = 60;
        #         viewer.timeline.zoomTo(clockStart, clockStop);

        #         var property = new Cesium.SampledPositionProperty();
        #         for (var i = 0; i < longitudes.length; i++) {{
        #             var t   = Cesium.JulianDate.addSeconds(clockStart, i, new Cesium.JulianDate());
        #             var pos = Cesium.Cartesian3.fromDegrees(longitudes[i], latitudes[i], altitudes[i]);
        #             property.addSample(t, pos);
        #         }}

        #         var satEntity = viewer.entities.add({{
        #             id: 'animated-satellite',
        #             name: '{satellite}',
        #             availability: new Cesium.TimeIntervalCollection([new Cesium.TimeInterval({{
        #                 start: clockStart,
        #                 stop:  clockStop
        #             }})]),
        #             position: property,
        #             model: {{
        #                 uri: 'examples/resources/sat/ICESat-2.glb',
        #                 minimumPixelSize: 128,
        #                 maximumScale: 10000
        #             }},
        #             label: {{
        #                 text: '{satellite}',
        #                 font: 'bold 16px sans-serif',
        #                 fillColor: Cesium.Color.BLUE,
        #                 style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        #                 outlineWidth: 2,
        #                 verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        #                 pixelOffset: new Cesium.Cartesian2(0, -20)
        #             }},
        #             path: {{
        #                 resolution: 1,
        #                 material: Cesium.Color.BLUE.withAlpha(0.7),
        #                 width: 4,
        #                 leadTime: 0,
        #                 trailTime: longitudes.length
        #             }}
        #         }});
                
        #         // ── Dynamic link lines: GS → satellite only during each pass ────────
        #         var gsPosition = Cesium.Cartesian3.fromDegrees({gs_lon}, {gs_lat}, {gs_alt});
        #         for (var p = 0; p < passIntervals.length; p++) {{
        #             (function(interval) {{
        #                 var passStart = Cesium.JulianDate.addSeconds(clockStart, interval.start_idx, new Cesium.JulianDate());
        #                 var passStop  = Cesium.JulianDate.addSeconds(clockStart, interval.end_idx,   new Cesium.JulianDate());
        #                 viewer.entities.add({{
        #                     availability: new Cesium.TimeIntervalCollection([new Cesium.TimeInterval({{
        #                         start: passStart,
        #                         stop:  passStop
        #                     }})]),
        #                     polyline: {{
        #                         positions: new Cesium.CallbackProperty(function(time, result) {{
        #                             var satPos = property.getValue(time, result);
        #                             if (!Cesium.defined(satPos)) return [gsPosition, gsPosition];
        #                             return [gsPosition, satPos];
        #                         }}, false),
        #                         width: 2,
        #                         material: new Cesium.PolylineGlowMaterialProperty({{
        #                             glowPower: 0.25,
        #                             color: Cesium.Color.CYAN.withAlpha(0.85)
        #                         }})
        #                     }}
        #                 }});
        #             }})(passIntervals[p]);
        #         }}
                

        #         viewer.zoomTo(satEntity);
        #     </script>
        # </body>
        # </html>
        # '''

        # Load Cesium HTML template (same pattern as plot_3d_trajectory)
        template_path = Path(self.datadir) / 'cesium_3d_ground_station.html'
        template = template_path.read_text()
        pass_start_positions = [[lon, lat, alt] for lon, lat, alt in zip(pass_start_lons, pass_start_lats, pass_start_alts)]
        pass_end_positions = [[lon, lat, alt] for lon, lat, alt in zip(pass_end_lons, pass_end_lats, pass_end_alts)]
        cesium_html = template.format(
            longitudes=json.dumps(list(longitudes)),
            latitudes=json.dumps(list(latitudes)),
            altitudes=json.dumps(list(altitudes)),
            pass_intervals=json.dumps(pass_intervals_js),
            pass_start_positions=json.dumps(pass_start_positions),
            pass_end_positions=json.dumps(pass_end_positions),
            selected_gs=json.dumps(selected_gs),
            gs_lon=gs_lon,
            gs_lat=gs_lat,
            gs_alt=gs_alt,
            satellite=json.dumps(satellite),
            time_iso=json.dumps(time_iso)
        )

        output_dir = os.path.join(self.outputdir, 'figures')
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, 'OGS_3D.html')
        with open(output_file_path, 'w') as f:
            f.write(cesium_html)
        print(f"3D groundstation visualization saved to {output_file_path}.")
        self.start_http_server_and_open(output_file_path)
    
    def run_orbit_simulation(self, hp_setting=0, sim_time=24*3600, pe_flag = False):
        """Run the orbit simulation"""
        # Get simulation parameters
        if not pe_flag:
            sim_time = self.sim_time.value() * 3600  # Convert hours to seconds
            hp_setting = self.precision.currentIndex()
            
        # Load satellite configuration
        config_path = os.path.join(os.path.dirname(__file__), self.datadir, 'satellite_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        sat_names = config['sat_names']
        n_sats = len(sat_names)
        
        # Initialize TUDAT simulation
        spice.load_standard_kernels()
        
        # Set simulation parameters
        # simulation_start_epoch = 25 * 365.25 * 24 * 3600  # Start from 2022
        
        # J2000 time
        t_j2000_now = t_conv.gps_to_j2000(t_conv.utc2gws(dt.datetime.now()))
        simulation_start_epoch = t_j2000_now
        
        simulation_end_epoch = simulation_start_epoch + sim_time
        
        if pe_flag:
            save_append = "pointing_error"
            output_dir = os.path.join(self.outputdir, save_append)
            state_history_path = os.path.join(output_dir, 'state_history.dat')
            if os.path.exists(state_history_path):
                print("state_history.dat already exists.")
                return output_dir
            else:
                fixed_step_size = 1
                os.makedirs(output_dir, exist_ok=True)
        
        else: 
            fixed_step_size = 60
            save_append = self.precision.currentText()            
            output_dir = os.path.join(self.outputdir, save_append)
            os.makedirs(output_dir, exist_ok=True)
        
        # Set Drag/SRP reference area, coefficients, mass
        mass = 400  # kg
        reference_area = 3  # m^3
        drag_coefficient = 2.0
        radiation_pressure_coefficient = 1.2
        
        # High precision settings
        if hp_setting == 1:
            radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
                "Sun", reference_area, radiation_pressure_coefficient, ['Earth']
            )
            aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
                reference_area, [drag_coefficient, 0, 0]
            )
        
        # Create acceleration model class
        acceleration_models_available = util.select_acceleration_model()
        
        # Set acceleration settings
        if hp_setting == 1: # High precision
            acceleration_setting_earth = 7
            acceleration_setting_3rdbody = [0,1]  # Add Moon and Sun PMG accelerations
            add_srp = 1
            integration_setting = "rk4"
        elif hp_setting == 0: # Medium precision
            acceleration_setting_earth = 6
            acceleration_setting_3rdbody = [0,1]  # Add Moon and Sun PMG accelerations
            add_srp = 0
            integration_setting = "rk4"
        else: # J2 only
            acceleration_setting_earth = 1  
            acceleration_setting_3rdbody = []
            add_srp = 0
            integration_setting = "rk4"
            
        acceleration_label, acceleration_settings_sat = acceleration_models_available.select_model(
            setting_earth=acceleration_setting_earth,
            setting_other_bodies=acceleration_setting_3rdbody,
            add_srp=add_srp
        )
        
        # Set up initial conditions from config
        ic_sats = np.array([
            [ic['a']*1e3, ic['e'], np.deg2rad(ic['i']), 
            np.deg2rad(ic['w']), np.deg2rad(ic['RAAN']), 
            np.deg2rad(ic['theta'])]
            for ic in config['initial_conditions']
        ])
        
        # Create bodies
        bodies_to_create = [body for body in acceleration_settings_sat.keys()]
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"
        
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation)
        
        bodies = environment_setup.create_system_of_bodies(body_settings)
        earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
        earth_average_radius = spice.get_average_radius("Earth")
        
        # Convert to Cartesian elements
        ic_const_all = np.zeros((n_sats, 6))
        for ii, row in enumerate(ic_const_all):
            ic_const_all[ii,:] = element_conversion.keplerian_to_cartesian_elementwise(
                gravitational_parameter=earth_gravitational_parameter,
                semi_major_axis=ic_sats[ii,0],
                eccentricity=ic_sats[ii,1],
                inclination=ic_sats[ii,2],
                argument_of_periapsis=ic_sats[ii,3],
                longitude_of_ascending_node=ic_sats[ii,4],
                true_anomaly=ic_sats[ii,5]
            )
        initial_states = np.concatenate(ic_const_all)
        
        # Setup constellation generation inputs
        bodies_to_propagate = sat_names
        acceleration_settings = {}
        for sat_name in sat_names:
            bodies.create_empty_body(sat_name)
            
            if hp_setting == 1:
                # Add SRP interface
                environment_setup.add_radiation_pressure_interface(
                    bodies, sat_name, radiation_pressure_settings
                )
                # Add Drag interface
                environment_setup.add_aerodynamic_coefficient_interface(
                    bodies, sat_name, aero_coefficient_settings)
                bodies.get(sat_name).mass = mass
            acceleration_settings[sat_name] = acceleration_settings_sat
        
        central_bodies = ["Earth" for ii in bodies_to_propagate]
        
        # Create acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )
        
        # Create numerical integrator settings
        if integration_setting == "rk4":
            integrator_settings = propagation_setup.integrator.runge_kutta_4(
                simulation_start_epoch, fixed_step_size
            )

        # Create 8th order RKF settings
        elif integration_setting == "rkf_78":
        
            # Define step-size control settings using scalar tolerances
            control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(
                relative_error_tolerance=1e-9,
                absolute_error_tolerance=1e-11
            )

            # Define validation settings (min/max step limits)
            validation_settings = propagation_setup.integrator.step_size_validation(
                minimum_step=1.0,
                maximum_step=300.0
            )

            # Create variable-step integrator settings
            integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
                initial_time_step=1.0,
                coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78,
                step_size_control_settings=control_settings,
                step_size_validation_settings=validation_settings,
                assess_termination_on_minor_steps=False
            )

        
        # Execute simulation
        dep_var_to_save = []
        termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_states,
            termination_condition,
            output_variables=dep_var_to_save
        )
        
        # Create simulation object and propagate dynamics
        # dynamics_simulator = dynamics.SingleArcSimulator(
        #     bodies, integrator_settings, propagator_settings,
        #     print_dependent_variable_data=False,
        #     print_state_data=False
        # )
        
        dynamics_simulator =  dynamics.SingleArcSimulator(
            bodies, integrator_settings, propagator_settings,
            print_dependent_variable_data=False,
            print_state_data=False
        )
        
        # Get results
        states = dynamics_simulator.state_history
        states_array = result2array(states)
        
        # Create output directory
        states_df = pd.DataFrame(states_array)
        states_df.to_csv(os.path.join(output_dir, 'state_history.dat'), sep='\t', index=False, header=False)
            
        # Create r_index dictionary for state indices
        r_index = {}
        for i, sat_name in enumerate(sat_names):
            r_index[sat_name] = [1 + 6*i, 2 + 6*i, 3 + 6*i]  # x,y,z indices for each satellite
        
        # Save simulation parameters with required structure
        simulation_parameters = {
            "name": save_append,
            "t_start": simulation_start_epoch,
            "t_end": simulation_end_epoch,
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "accelerations_setting": acceleration_label,
            "integrator": integration_setting,
            "time_step": fixed_step_size,
            "constellation": {},
            "total_sats": n_sats,
            "simulation_run_time": 0.0,  # This would need to be measured
            "n_function_evaluations": int((simulation_end_epoch - simulation_start_epoch) / fixed_step_size),
            "sat_names": sat_names,
            "r_index": r_index
        }
        self.update_sim_info_text(simulation_parameters)
        
        with open(os.path.join(output_dir, 'simulation_parameters.json'), 'w') as f:
            json.dump(simulation_parameters, f, indent=4)
        
        # Store results for GUI
        self.simulation_data = {
            'states': states_array,
            'sat_names': sat_names,
            'time': states_array[:, 0]
        }
        
        if pe_flag:
            return output_dir
        # Populate satellite dropdowns
        self.load_and_populate_sat_names(output_dir)
        
        # Plot trajectories
        self.plot_trajectories()
        self.plot_3d_trajectories()
        
    def generate_attitude_selector(self):
        selected_interpolator = self.active_interpolator
        selected_integrator   = self.active_integrator
        
        if selected_interpolator == "CUBIC-SPLINE":
            print(f'''Attitude generation:
                Rotation: Euler angles 
                Integration method: Explicit Euler
                Interpolation method: {selected_interpolator }
                Nonlinear coupling: Ignored
                Large rotations: Underestimated
                Overall: Poor
                ''')
            self.generate_attitude()
        else:
            print(f'''Attitude generation:
                Rotation: Quaternions
                Integration method: {selected_integrator}
                Interpolation method: {selected_interpolator}
                Nonlinear coupling: Preserved
                Large rotations: Correct
                Overall: Good
                
                ''')
            self.generate_attitude_new()        
        
    def generate_attitude_new(self):
        """Generate attitude data based on selected settings from JSON, with conditional logic"""
        # Load settings from JSON file
        settings_file = os.path.join(os.path.dirname(__file__), 'input_data','attitude_settings.json')
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        
        # Get selected settings
        selected_settings_name = self.settings_combo.currentText()
        selected_settings = settings[selected_settings_name]
        
        # Get attitude parameters from UI
        roll = self.roll.value()
        pitch = self.pitch.value()
        yaw = self.yaw.value()
        
        # Set up parameters from JSON in rad
        omega_vec = np.array(selected_settings['omega_vec'])
        omega_dot_vec = np.array(selected_settings['omega_dot_vec'])
        omega_dot_swapped_vec = - omega_dot_vec

        # Add any additional parameters if present in JSON, e.g., om_ddot_vec if defined
        if 'omega_ddot_vec' in selected_settings:
            om_ddot_vec = np.array(selected_settings['omega_ddot_vec'])
        else:
            om_ddot_vec = np.zeros(3)  # Default if not in JSON
        
        # Apply normalization if specified in JSON
        if selected_settings.get('normalize', False):
            omega_vec /= np.sqrt(3)
            omega_dot_vec /= np.sqrt(3)
            if 'om_ddot_vec' in selected_settings:
                om_ddot_vec /= np.sqrt(3)

        #att_ea_0 = np.array([roll, pitch, yaw])  # Using your manual change
        t_0 = selected_settings['t_0']
        t_step = selected_settings['t_step']
        t_end = selected_settings['t_end']
        t_vec_prop = np.arange(t_0, t_end + t_step, t_step)
        
        nrows = len(t_vec_prop)
        
        ea_all = np.zeros((nrows, 3))
        ea_dot_all = np.zeros((nrows, 3))
        omega_vec_all = np.zeros((nrows, 3))
        omega_dot_vec_all = np.zeros((nrows, 3))  # For tracking if needed
        omega_ddot_vec_all = np.zeros((nrows, 3))  # For tracking if needed
        quat_all = np.zeros((nrows, 4))
        quat_rate_all = np.zeros((nrows, 4))

        # Limits and timing
        alpha_limit = np.deg2rad(np.linalg.norm(omega_dot_vec))
        dt_hold1 = 1.5
        dt_hold2 = 0.5
        t_hold_switch = 50.0

        # Initial conditions
        q = quat_slerp.quat_from_euler_deg(roll, pitch, yaw)
        q = quat_slerp.normalize(q)

        omega = np.deg2rad(omega_vec)
        alpha = np.deg2rad(omega_dot_vec)
        jerk0  = np.deg2rad(om_ddot_vec)

        quat_all[0] = q
        quat_rate_all[0] = quat_slerp.quaternion_derivative(q, omega)
        omega_vec_all[0] = omega
        omega_dot_vec_all[0] = alpha
        N = len(t_vec_prop)
        jerk_history  = np.zeros((N, 3))
        jerk = jerk0.copy()

        if 'rocketlab_march' in selected_settings_name:
            # State machine flags
            hold_active = False
            t_hold_end = None
            acc_go_mode = True

            for ii, t in enumerate(t_vec_prop):
                if ii == 0:
                    jerk_history[0] = jerk
                    continue
                # ----------------------------------------------------------
                # 1) JERK / HOLD STATE MACHINE  (RocketLab logic)
                # ----------------------------------------------------------

                acc_mag = np.linalg.norm(alpha)   ### FIXED ###
                #alpha_limit = np.deg2rad(alpha_mag)

                if acc_mag >= alpha_limit:  # ← FIXED: >= not 
                    if t > t_hold_switch:
                        hold_duration = dt_hold2
                    else:
                        hold_duration = dt_hold1
                        
                    if acc_go_mode:
                        hold_active = True
                        
                    if acc_go_mode and hold_active:
                        t_hold_end = t + hold_duration
                        acc_go_mode = False
                    
                    if t <= t_hold_end:
                        jerk = np.zeros(3)
                    else:
                        jerk = -np.sign(alpha[0]) * jerk0  # ← FIXED: use jerk0, not unit vector
                        acc_go_mode = True
                        hold_active = False
                #else:
                #    jerk = np.zeros(3)

                # ----------------------------------------------------------
                # 2) RK4 PROPAGATION
                # ----------------------------------------------------------
                if self.active_integrator == 'RK4':
                    q, q_dot, omega, alpha = quat_slerp.rk4_step(q, omega, alpha, jerk, t_step)
                elif self.active_integrator == 'Rapid 4th-Order':
                    q, q_dot, omega, alpha = quat_slerp.integrate_high_order_aocs(q, omega, alpha, jerk, t_step, norm_threshold=1e-6)

                # ----------------------------------------------------------
                # 3) SAVE STATES
                # ----------------------------------------------------------
                quat_all[ii]       = q
                quat_rate_all[ii]   = q_dot
                omega_vec_all[ii]   = omega
                #alpha_history[ii]   = alpha
                jerk_history[ii]    = jerk 

        else:
            # Reset initial conditions
            alpha_initial = alpha.copy()
            alpha_swapped  = -alpha_initial # rad/s²
            jerk_current   = np.zeros(3)  # jerk = alpha_dot

            swap_time = 40.0
            # jerk_flag = True

            # RK4 loop
            for ii, t in enumerate(t_vec_prop[1:], start=1):
                # Swap angular acceleration at t >= 40 s
                
                if t >= swap_time and self.jerk_flag:
                    alpha_current = alpha_swapped
                else:
                    alpha_current = alpha_initial

                # RK4 step (jerk = 0)
                q, q_dot, omega, _ =  quat_slerp.rk4_step(q, omega, alpha_current, np.zeros(3), t_step)

                # Store
                quat_all[ii]       = q
                quat_rate_all[ii]   = q_dot
                omega_vec_all[ii]   = omega
                # alpha_history[ii]   = alpha
                # jerk_history[ii]    = jerk_current  

        ea_all      = np.vstack([quat_slerp.euler_from_quat(q) for q in quat_all]) 
        ea_dot_all  = quat_slerp.euler_rates_321(ea_all, omega_vec_all)
        self.attitude_data = {
            'time': t_vec_prop,
            'euler_angles': ea_all,
            'euler_rates': ea_dot_all,
            'angular_velocities': omega_vec_all,
            'quaternions': quat_all,
            'quaternion_rates': quat_rate_all
        }
        
        # Save results and update visualization as before
        output_dir = os.path.join(self.outputdir, 'tables', f'{selected_settings_name}_quatpred')
        os.makedirs(output_dir, exist_ok=True)
        
        self.quat_df = pd.DataFrame({
            'time': t_vec_prop,
            'q_w': quat_all[:, 0],
            'q_x': quat_all[:, 1],
            'q_y': quat_all[:, 2],
            'q_z': quat_all[:, 3],
            'q_w_dot': quat_rate_all[:,0],
            'q_x_dot': quat_rate_all[:,1],
            'q_y_dot': quat_rate_all[:,2],
            'q_z_dot': quat_rate_all[:,3]
        })
        
        output_file = os.path.join(output_dir, f'true_quat_{selected_settings_name}.csv')
        self.quat_df.to_csv(output_file, index=False)
        
        net_angle = quat_slerp.quat_angle(quat_all[-1])        
        print("="*60)
        print(f"Net rotation from start to end: {net_angle:.3f} deg")
        print(f'''
        Final roll, pitch, yaw : {ea_all[-1]} [deg] 
        Final angular velocity : {np.rad2deg(omega_vec_all[-1])} [deg/s]
        MAX angular velocity : {np.rad2deg(np.max(np.linalg.norm(omega_vec_all, axis = 1))):.3f} [deg/s]
        constant angular rate : {omega_dot_vec} [deg/s^2]
        --------------------------------------------------
        Calculated attitude kinematics for {t_end} s, with {t_step} steps
        --------------------------------------------------
        ''')

        # Switch to the Attitude Visualization tab in the graphics panel
        att_tab_index = self.graphics_tabs.indexOf(self.attitude_graphics)
        if att_tab_index != -1:
            self.graphics_tabs.setCurrentIndex(att_tab_index)
        self.update_attitude_visualization()

    def generate_attitude(self):
        """Generate attitude data based on selected settings from JSON, with conditional logic"""
        # Load settings from JSON file
        settings_file = os.path.join(os.path.dirname(__file__), 'input_data','attitude_settings.json')
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        
        # Get selected settings
        selected_settings_name = self.settings_combo.currentText()
        selected_settings = settings[selected_settings_name]
        
        # Get attitude parameters from UI
        roll = self.roll.value()
        pitch = self.pitch.value()
        yaw = self.yaw.value()
        
        # Set up parameters from JSON
        omega_vec = np.array(selected_settings['omega_vec'])
        omega_dot_vec = np.array(selected_settings['omega_dot_vec'])
        omega_dot_swapped_vec = - omega_dot_vec

        # Add any additional parameters if present in JSON, e.g., om_ddot_vec if defined
        if 'omega_ddot_vec' in selected_settings:
            om_ddot_vec = np.array(selected_settings['omega_ddot_vec'])
        else:
            om_ddot_vec = np.zeros(3)  # Default if not in JSON
        
        # Apply normalization if specified in JSON
        if selected_settings.get('normalize', False):
            omega_vec /= np.sqrt(3)
            omega_dot_vec /= np.sqrt(3)
            if 'om_ddot_vec' in selected_settings:
                om_ddot_vec /= np.sqrt(3)
        
        att_ea_0 = np.array([roll, pitch, yaw])  # Using your manual change
        t_0 = selected_settings['t_0']
        t_step = selected_settings['t_step']
        t_end = selected_settings['t_end']
        t_vec_prop = np.arange(t_0, t_end + t_step, t_step)
        
        nrows = len(t_vec_prop)
        ea_all = np.zeros((nrows, 3))
        ea_dot_all = np.zeros((nrows, 3))
        omega_vec_all = np.zeros((nrows, 3))
        omega_dot_vec_all = np.zeros((nrows, 3))  # For tracking if needed
        omega_ddot_vec_all = np.zeros((nrows, 3))  # For tracking if needed
        quat_all = np.zeros((nrows, 4))
        quat_rate_all = np.zeros((nrows, 4))

        
        if 'rocketlab_march' in selected_settings_name:  # Adaptive logic from example_att_generation.py
            om_dot_limit = selected_settings.get('om_dot_limit', 0.0837)  # From JSON
            t_hold_scenario_end = selected_settings.get('t_hold_scenario_end', 50)
            dt_hold = selected_settings.get('dt_hold', 1.5)

            acc_go_mode = 1
            acc_hold_mode = 0
            ea_0 = att_ea_0
            om_0 = omega_vec
            om_dot_0 = omega_dot_vec
            om_ddot_0 = om_ddot_vec  # Use from JSON
            
            for ii, t_ii in enumerate(t_vec_prop):
                if ii == 0:
                    ## initialize
                    ea_0 = att_ea_0
                    om_0 = omega_vec
                    om_dot_0 = omega_dot_vec*1
                    om_ddot_0 = om_ddot_vec[0]
                # Add jerk motion
                if t_ii >= 40 and self.jerk_flag:
                    omega_dot_vec = -omega_dot_vec
                    # print(f't = {t_ii} We swapping accelerations to {omega_dot_vec} deg/s^2')

                if np.linalg.norm(om_dot_0) >= om_dot_limit:
                    if t_ii > t_hold_scenario_end:
                        dt_hold = 0.5# selected_settings.get('dt_hold_adjusted', dt_hold)  # Adjusted from JSON if available
                    if acc_go_mode : #and not acc_hold_mode:
                        acc_hold_mode = 1
                    if acc_go_mode and acc_hold_mode:
                        t_0_local = t_ii
                        t_f_local = t_ii + dt_hold
                        acc_go_mode = 0
                    if t_ii <= t_f_local:
                        om_ddot_0 = np.zeros(3)  # Hold rate
                    else:
                        om_ddot_0 = -np.sign(om_dot_0[0]) * om_ddot_vec  # Swap based on sign
                        acc_go_mode = 1
                        acc_hold_mode = 0
                
                om_ii = om_0 + t_step * om_dot_0
                om_dot_ii = om_dot_0 + t_step * om_ddot_0
                ea_rate = att_conv.calc_ea_dot(ea_0, om_ii, deg=1)
                ea_ii = ea_0 + t_step * ea_rate
                quat_ii, quat_rate_ii = att_conv.calc_qdot(ea_ii, om_ii, deg=1)
                quat_ii = quat_ii.flatten()
                quat_rate_ii = quat_rate_ii.flatten()
                
                ea_all[ii,:] = ea_0
                ea_dot_all[ii,:] = ea_rate
                omega_vec_all[ii,:] = om_ii
                omega_dot_vec_all[ii,:] = om_dot_ii
                omega_ddot_vec_all[ii,:] = om_ddot_0
                quat_all[ii,:] = quat_ii
                quat_rate_all[ii,:] = quat_rate_ii
                
                ea_0 = ea_ii
                om_dot_0 = om_dot_ii
                om_0 = om_ii
        else:  # Default logic for other settings
            ea_0 = att_ea_0
            om_0 = omega_vec
            for ii, t_ii in enumerate(t_vec_prop):
                # Add jerk
                if t_ii >= 40 and self.jerk_flag:
                    omega_dot_vec = omega_dot_swapped_vec
                    # print(f't = {t_ii} We swapping accelerations to {omega_dot_vec} deg/s^2')

                om_ii = om_0 + t_step * omega_dot_vec
                ea_rate = att_conv.calc_ea_dot(ea_0, om_ii)
                ea_ii = ea_0 + t_step * ea_rate
                quat_ii, quat_rate_ii = att_conv.calc_qdot(ea_ii, om_ii, deg=1)
                quat_ii = quat_ii.flatten()
                quat_rate_ii = quat_rate_ii.flatten()
                
                ea_all[ii,:] = ea_0
                ea_dot_all[ii,:] = ea_rate
                omega_vec_all[ii,:] = om_ii
                quat_all[ii,:] = quat_ii
                quat_rate_all[ii,:] = quat_rate_ii
                
                ea_0 = ea_ii
                om_0 = om_ii
        
        self.attitude_data = {
            'time': t_vec_prop,
            'euler_angles': ea_all,
            'euler_rates': ea_dot_all,
            'angular_velocities': omega_vec_all,
            'quaternions': quat_all,
            'quaternion_rates': quat_rate_all
        }
        
        # Save results and update visualization as before
        output_dir = os.path.join(self.outputdir, 'tables', f'{selected_settings_name}_quatpred')
        os.makedirs(output_dir, exist_ok=True)
        
        self.quat_df = pd.DataFrame({
            'time': t_vec_prop,
            'q_w': quat_all[:, 0],
            'q_x': quat_all[:, 1],
            'q_y': quat_all[:, 2],
            'q_z': quat_all[:, 3],
            'q_w_dot': quat_rate_all[:,0],
            'q_x_dot': quat_rate_all[:,1],
            'q_y_dot': quat_rate_all[:,2],
            'q_z_dot': quat_rate_all[:,3]
        })
        
        output_file = os.path.join(output_dir, f'true_quat_{selected_settings_name}.csv')
        self.quat_df.to_csv(output_file, index=False)
        
        print(f'''                  DONE
        Final roll, pitch, yaw : {ea_ii} [deg] 
        Final angular velocity : {om_ii} [deg/s]
        MAX angular velocity : {np.max(np.linalg.norm(omega_vec_all, axis = 1)):.3f} [deg/s]
        constant angular rate : {omega_dot_vec} [deg/s^2]
        --------------------------------------------------
        Calculated attitude kinematics for {t_end} s, with {t_step} steps
        --------------------------------------------------
        ''')

        # Switch to the Attitude Visualization tab in the graphics panel
        att_tab_index = self.graphics_tabs.indexOf(self.attitude_graphics)
        if att_tab_index != -1:
            self.graphics_tabs.setCurrentIndex(att_tab_index)
        self.update_attitude_visualization()
        
    def update_attitude_visualization(self):
        """Update the attitude visualization plot"""
        if not self.attitude_data:
            print("Error: self.attitude_data is empty or not set.")
            return
            
        # --- DEBUG: Set custom page for JS error logging ---
        self.attitude_plot.setPage(DebugWebEnginePage(self.attitude_plot))
            
        # Verify data shapes for debugging
        try:
            if np.any(np.isnan(self.attitude_data['euler_angles'])) or np.any(np.isnan(self.attitude_data['euler_rates'])):
                print("Error: self.attitude_data contains NaN values.")
        except Exception as e:
            print(f"Error checking data: {e}")
            
        # Downsample for GUI if needed
        max_points = 1000
        total_points = len(self.attitude_data['time'])
        if total_points > max_points:
            idx = np.linspace(0, total_points - 1, max_points).astype(int)
            time = self.attitude_data['time'][idx]
            euler_angles = self.attitude_data['euler_angles'][idx, :]
            euler_rates = self.attitude_data['euler_rates'][idx, :]
            angular_velocities = self.attitude_data['angular_velocities'][idx, :]
            quaternions = self.attitude_data['quaternions'][idx, :]
            downsample_note = f" (showing {max_points} of {total_points} points)"
        else:
            time = self.attitude_data['time']
            euler_angles = self.attitude_data['euler_angles']
            euler_rates = self.attitude_data['euler_rates']
            angular_velocities = self.attitude_data['angular_velocities']
            quaternions = self.attitude_data['quaternions']
            downsample_note = ""

        # Create plotly figure with 2 rows and 2 columns
        fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(
                            'Euler Angles', 'Angular Velocities',
                            'Euler Rates', 'Quaternions'),
                        horizontal_spacing=0.12,
                        vertical_spacing=0.12)
        
        # Add Euler angles (row 1, col 1)
        angles = ['Roll', 'Pitch', 'Yaw']
        for i, angle in enumerate(angles):
            fig.add_trace(
                go.Scatter(x=time,
                        y=(euler_angles[:,i]),
                        name=angle),
                row=1, col=1
            )
        
        # Add angular velocities (row 1, col 2)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            fig.add_trace(
                go.Scatter(x=time,
                        y=(angular_velocities[:,i]),
                        name=f'ω{axis}'),
                row=1, col=2
            )
        
        # Add Euler rates (row 2, col 1)
        for i, angle in enumerate(angles):
            fig.add_trace(
                go.Scatter(x=time,
                        y=(euler_rates[:,i]),
                        name=f'{angle} Rate'),
                row=2, col=1
            )
        
        # Add quaternions (row 2, col 2)
        for i, comp in enumerate(['w', 'x', 'y', 'z']):
            fig.add_trace(
                go.Scatter(x=time,
                        y=quaternions[:,i],
                        name=f'q{comp}'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Attitude Visualization{downsample_note}",
            title_x=0.5
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time [s]", row=1, col=1)
        fig.update_xaxes(title_text="Time [s]", row=1, col=2)
        fig.update_xaxes(title_text="Time [s]", row=2, col=1)
        fig.update_xaxes(title_text="Time [s]", row=2, col=2)
        fig.update_yaxes(title_text="Angle [deg]", row=1, col=1)
        fig.update_yaxes(title_text="ω [deg/s]", row=1, col=2)
        fig.update_yaxes(title_text="Rate [deg/s]", row=2, col=1)
        fig.update_yaxes(title_text="Quaternion", row=2, col=2)
        
        # Convert to HTML and display in GUI (downsampled)
        html = fig.to_html(include_plotlyjs='cdn')
        self.attitude_plot.setHtml(html)
        print("Attitude Plotly HTML (downsampled for GUI) set for attitude_plot.")

        # Save the full-resolution figure to a dedicated folder (for browser viewing)
        fig_full = make_subplots(rows=2, cols=2,
                        subplot_titles=(
                            'Euler Angles', 'Angular Velocities',
                            'Euler Rates', 'Quaternions'),
                        horizontal_spacing=0.12,
                        vertical_spacing=0.12)
        for i, angle in enumerate(angles):
            fig_full.add_trace(
                go.Scatter(x=self.attitude_data['time'],
                        y=(self.attitude_data['euler_angles'][:,i]),
                        name=angle),
                row=1, col=1
            )
        for i, axis in enumerate(['X', 'Y', 'Z']):
            fig_full.add_trace(
                go.Scatter(x=self.attitude_data['time'],
                        y=(self.attitude_data['angular_velocities'][:,i]),
                        name=f'ω{axis}'),
                row=1, col=2
            )
        for i, angle in enumerate(angles):
            fig_full.add_trace(
                go.Scatter(x=self.attitude_data['time'],
                        y=(self.attitude_data['euler_rates'][:,i]),
                        name=f'{angle} Rate'),
                row=2, col=1
            )
        for i, comp in enumerate(['w', 'x', 'y', 'z']):
            fig_full.add_trace(
                go.Scatter(x=self.attitude_data['time'],
                        y=self.attitude_data['quaternions'][:,i],
                        name=f'q{comp}'),
                row=2, col=2
            )
        fig_full.update_layout(
            height=800,
            showlegend=True,
            title_text="Attitude Visualization (Full Resolution)",
            title_x=0.5
        )
        fig_full.update_xaxes(title_text="Time [s]", row=1, col=1)
        fig_full.update_xaxes(title_text="Time [s]", row=1, col=2)
        fig_full.update_xaxes(title_text="Time [s]", row=2, col=1)
        fig_full.update_xaxes(title_text="Time [s]", row=2, col=2)
        fig_full.update_yaxes(title_text="Angle [deg]", row=1, col=1)
        fig_full.update_yaxes(title_text="ω [deg/s]", row=1, col=2)
        fig_full.update_yaxes(title_text="Rate [deg/s]", row=2, col=1)
        fig_full.update_yaxes(title_text="Quaternion", row=2, col=2)
        output_dir = os.path.join(self.outputdir, 'figures')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'attitude_visualization.html')
        with open(output_file, 'w') as f:
            f.write(fig_full.to_html(include_plotlyjs='cdn'))
        print(f"Full-resolution figure saved successfully to {output_file}")



    def normalize(self,v):
        return v/np.linalg.norm(v)
    
    def analyze_link(self):
        """Analyze link between satellites"""
        if not self.simulation_data:
            return
            
        # Get selected satellites
        host_sat = self.host_sat.currentText()
        target_sat = self.target_sat.currentText()
        
        # Get satellite states
        host_idx = self.simulation_data['sat_names'].index(host_sat)
        target_idx = self.simulation_data['sat_names'].index(target_sat)
        
        t_j2000 = self.simulation_data['time']
        r_host = self.simulation_data['states'][:, 1+6*host_idx:4+6*host_idx]
        v_host = self.simulation_data['states'][:, 4+6*host_idx:7+6*host_idx]
        r_target = self.simulation_data['states'][:, 1+6*target_idx:4+6*target_idx]
        
        # Calculate RSW to ECI rotation matrix
        tudconv = tudatconv.tudat_predictor()
        ROT_RSWfromECI = np.array([tudconv.calc_rotrsweci(r_h=r_host[ii,:], v_h=v_host[ii,:]) 
                                for ii, r in enumerate(r_host)])
        
        # Convert to quaternions
        quat_eci2rsw = np.array([att_conv.convert_dcm2quat(dcm_ii) 
                                for dcm_ii in ROT_RSWfromECI])
        
        # Calculate azimuth, elevation, and range
        aer = ae_calc.calc_ae_full(r_host, r_target, attitude_eci2bf=quat_eci2rsw,
                                check_occultation=0)  # rad rad m
        
        # Store results
        self.link_data = {
            'time': t_j2000,
            'azimuth': aer[:,0],
            'elevation': aer[:,1],
            'range': aer[:,2]
        }
        
        # Update visualization
        self.update_link_visualization()

    def orbit_prop(self, t, vec_sv):
        # Constants
        mu = 398600.44  # km^3/s^2
        J2 = 1082.6267e-6
        R = 6378.1366  # km, Earth Radius
        
        r0 = vec_sv[:3]
        v0 = vec_sv[3:6]
        r_norm = max(np.linalg.norm(r0),1e-9)
        a1 = -(mu / r_norm**3) * r0
        
        # J2 Perturbations
        const = -3 * J2 * mu * R**2 / (2 * r_norm**5)
        
        ai = const *r0[0]* (5 * r0[2]**2 / r_norm**2 - 1)
        aj = const *r0[1]* (5 * r0[2]**2 / r_norm**2 - 1)
        ak = const *r0[2]* (5 * r0[2]**2 / r_norm**2 - 3)
        acc = np.array([ai, aj, ak])
        a = a1 + acc
        
        return np.hstack([v0, a])

    def odeRK(self, forbit, tspan, x0,substeps=20):
        """Improved RK4 with internal sub-stepping for stability"""
        N = len(tspan)
        n = len(x0)
        x0 = x0.reshape(-1, 1)
        x = np.zeros((N, n))
        x[0, :] = x0.flatten()
        w = x0.flatten()

        for i in range(N-1):
            h_outer = tspan[i+1] - tspan[i]          # usually 1.0 s
            h = h_outer / substeps                   # smaller internal step

            t = tspan[i]
            for _ in range(substeps):
                K1 = h * forbit(t, w)
                K2 = h * forbit(t + h/2, w + K1/2)
                K3 = h * forbit(t + h/2, w + K2/2)
                K4 = h * forbit(t + h, w + K3)
                w = w + (K1 + 2*K2 + 2*K3 + K4) / 6
                t += h

            x[i+1, :] = w

        return tspan, x

    def odeRK_single(self, forbit, tspan, x0):
        tspan = np.array(tspan, dtype=np.float32)
        x0 = np.array(x0, dtype=np.float32)
        N = len(tspan)
        n = len(x0)
        x0 = x0.reshape(-1, 1)
        x = np.zeros((N, n), dtype=np.float32)
        x[0, :] = x0.flatten()
        w = x0.flatten()
        for i in range(N-1):
            h = tspan[i+1] - tspan[i]
            t = tspan[i]
            K1 = h * forbit(t, w)
            K2 = h * forbit(t + h/2, w + K1/2)
            K3 = h * forbit(t + h/2, w + K2/2)
            K4 = h * forbit(t + h, w + K3)
            w = w + (K1 + 2*K2 + 2*K3 + K4) / 6
            x[i+1, :] = w
        return tspan, x  
        
    def update_link_visualization(self):
        """Update the link analysis visualization"""
        if not self.link_data:
            return
            
        # Create plotly figure
        fig = make_subplots(rows=3, cols=1,
                        subplot_titles=('Azimuth', 'Elevation', 'Range'))
        
        # Add azimuth
        fig.add_trace(
            go.Scatter(x=self.link_data['time']/60,
                    y=np.rad2deg(self.link_data['azimuth']),
                    name='Azimuth'),
            row=1, col=1
        )
        
        # Add elevation
        fig.add_trace(
            go.Scatter(x=self.link_data['time']/60,
                    y=np.rad2deg(self.link_data['elevation']),
                    name='Elevation'),
            row=2, col=1
        )
        
        # Add range
        fig.add_trace(
            go.Scatter(x=self.link_data['time']/60,
                    y=self.link_data['range']/1000,  # Convert to km
                    name='Range'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True)
        fig.update_yaxes(title_text="Angle [deg]", row=1, col=1)
        fig.update_yaxes(title_text="Angle [deg]", row=2, col=1)
        fig.update_yaxes(title_text="Range [km]", row=3, col=1)
        fig.update_xaxes(title_text="Time [min]", row=3, col=1)
        
        # Convert to HTML and display
        html = fig.to_html(include_plotlyjs='cdn')
        self.link_plot.setHtml(html)

    def plot_ground_station_visibility(self, selected_gs, selected_satellite,
                                   pass_durations, pass_starts, pass_ends,
                                   ground_track, times,gimbal_az ,gimbal_el,
                                   elevation_angles, azimuth_angles, passes,
                                   tle_line1, tle_line2):

        from plotly.express.colors import qualitative

        MIN_ELEV = 20.0

        # ── Shared setup (unchanged from original) ────────────────────────────────
        longitudes, latitudes = np.array(ground_track).T

        def time_to_index(t):
            dt = (times - t).sec
            return int(np.argmin(np.abs(dt)))

        start_indices = [time_to_index(t) for t in pass_starts]
        end_indices   = [time_to_index(t) for t in pass_ends]

        # ── Per-pass color palette ────────────────────────────────────────────────
        palette = qualitative.Plotly + qualitative.Dark24 + qualitative.Light24
        colors  = [palette[i % len(palette)] for i in range(len(passes))]

        # ── TLE-derived parameters ────────────────────────────────────────────────
        raan_deg    = float(tle_line2[17:25])
        mean_motion = float(tle_line2[52:63])
        period_min  = 1440.0 / mean_motion
        window_h    = (times[-1] - times[0]).to(u.hour).value
        if len(pass_starts) > 1:
            gaps_min = [(pass_starts[i+1] - pass_starts[i]).to(u.minute).value
                        for i in range(len(pass_starts) - 1)]
            mean_revisit_min = float(np.mean(gaps_min))
        else:
            mean_revisit_min = 0.0

        # ── Hours axis for elevation plot ─────────────────────────────────────────
        t0    = times[0]
        hours = np.array([(t - t0).to(u.hour).value for t in times])

        # ════════════════════════════════════════════════════════════════════════
        # make_subplots — 4 rows (2 new + 2 original)
        # ════════════════════════════════════════════════════════════════════════
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                (f"Elevation from {selected_gs}  |  RAAN = {raan_deg:.3f}°  |  "
                f"T = {period_min:.2f} min  |  Window = {window_h:.1f} h  |  "
                f"Passes > {MIN_ELEV}° = {len(passes)}  |  "
                f"Mean revisit = {mean_revisit_min:.1f} min"),
                "Pass Durations",
                f"Sky Plot from {selected_gs}",
                "Satellite Ground Track",
            ),
            specs=[
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "polar"}],
                [{"type": "geo"}],
            ],
            row_heights=[0.22, 0.12, 0.28, 0.38],
            vertical_spacing=0.055
        )

        # ════════════════════════════════════════════════════════════════════════
        # ROW 1 — Elevation timeline  [NEW]
        # ════════════════════════════════════════════════════════════════════════
        fig.add_trace(go.Scatter(
            x=hours, y=elevation_angles,
            fill='tozeroy',
            fillcolor='rgba(70,130,180,0.35)',
            line=dict(color='steelblue', width=0.6),
            name='Elevation',
            showlegend=False,
            hovertemplate='%{x:.2f} h  |  %{y:.1f}°<extra></extra>'
        ), row=1, col=1)
        # ── Condor Gimbal Az/El overlay
        if gimbal_az is not None and gimbal_el is not None:
            fig.add_trace(go.Scatter(
                x=hours, y=gimbal_el,
                mode='lines',
                line=dict(color='orange', width=1.5),
                name='LVLH_el',
                showlegend=True,
                hovertemplate='%{x:.2f} h  |  %{y:.1f}°<extra></extra>'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=hours, y=gimbal_az,
                mode='lines',
                line=dict(color='green', width=1.5),
                name='LVLH_az',
                showlegend=True,
                hovertemplate='%{x:.2f} h  |  %{y:.1f}°<extra></extra>'
            ), row=1, col=1)

        # Threshold lines
        for y_val, col_str, dash in [(MIN_ELEV, 'red', 'dash'), (0.0, 'black', 'dash')]:
            fig.add_shape(type='line',
                        x0=0, x1=window_h, y0=y_val, y1=y_val,
                        line=dict(color=col_str, dash=dash, width=1.2),
                        row=1, col=1)

        # Per-pass peak markers + annotations
        for idx, p in enumerate(passes):
            si, ei   = p['start_idx'], p['end_idx']
            max_elev = p['max_elev']
            dur_min  = p['duration_min']
            if max_elev < MIN_ELEV:
                continue
            peak_i = si + int(np.argmax(elevation_angles[si:ei + 1]))
            t_peak = float(hours[peak_i])
            color  = colors[idx]

            fig.add_trace(go.Scatter(
                x=[t_peak], y=[max_elev],
                mode='markers',
                marker=dict(color=color, size=9, symbol='circle'),
                showlegend=False,
                hovertemplate=(f"Pass {idx+1}<br>{dur_min:.2f} min > {MIN_ELEV}°"
                            f"<br>Max el: {max_elev:.1f}°<extra></extra>")
            ), row=1, col=1)

            fig.add_annotation(
                x=t_peak, y=max_elev,
                row=1, col=1,
                text=f"{dur_min:.2f} min > {MIN_ELEV:.0f}°",
                showarrow=False, yshift=11,
                font=dict(size=8, color=color),
                bgcolor='rgba(255,255,255,0.65)'
            )

        fig.update_xaxes(title_text="Hours from start [h]", row=1, col=1,
                        range=[0, window_h], dtick=6, showgrid=True, gridcolor='rgba(0,0,0,0.12)')
        fig.update_yaxes(title_text="Elevation [deg]", row=1, col=1,
                        showgrid=True, gridcolor='rgba(0,0,0,0.12)')

        # ════════════════════════════════════════════════════════════════════════
        # ROW 2 — Pass durations bar chart  [ORIGINAL — unchanged]
        # ════════════════════════════════════════════════════════════════════════
        fig.add_trace(
            go.Bar(
                x=[f"Pass {i+1}" for i in range(len(pass_durations))],
                y=pass_durations,
                marker_color='blue',
                name="Pass Duration (min)"
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Duration (minutes)", row=2, col=1)
        fig.update_xaxes(title_text="Pass Number", row=2, col=1)

        # ════════════════════════════════════════════════════════════════════════
        # ROW 3 — Sky plot  [NEW]
        # ════════════════════════════════════════════════════════════════════════
        for idx, p in enumerate(passes):
            si, ei  = p['start_idx'], p['end_idx']
            elev_p  = elevation_angles[si:ei + 1]
            az_p    = azimuth_angles[si:ei + 1]
            mask    = elev_p >= MIN_ELEV
            if not np.any(mask):
                continue

            r_vals = 90.0 - elev_p[mask]   # zenith at centre
            az_sel = az_p[mask]
            color  = colors[idx]

            # Pass arc
            fig.add_trace(go.Scatterpolar(
                r=r_vals, theta=az_sel,
                mode='lines',
                line=dict(color=color, width=1.8),
                name=f'Pass {idx+1}',
                legendgroup=f'pass_{idx}',
                showlegend=True,
                hovertemplate=(f'Pass {idx+1}<br>Az: %{{theta:.1f}}°'
                            '<br>El: %{customdata:.1f}°<extra></extra>'),
                customdata=elev_p[mask]
            ), row=3, col=1)

            # AOS (×)
            fig.add_trace(go.Scatterpolar(
                r=[r_vals[0]], theta=[az_sel[0]],
                mode='markers',
                marker=dict(size=9, color=color, symbol='x'),
                showlegend=False, legendgroup=f'pass_{idx}'
            ), row=3, col=1)

            # Peak elevation (●)
            peak_local = int(np.argmin(r_vals))
            fig.add_trace(go.Scatterpolar(
                r=[r_vals[peak_local]], theta=[az_sel[peak_local]],
                mode='markers',
                marker=dict(size=9, color=color, symbol='circle'),
                showlegend=False, legendgroup=f'pass_{idx}'
            ), row=3, col=1)

        fig.update_polars(
            angularaxis=dict(
                direction='clockwise',
                rotation=90,                      # 0° = North at top
                tickvals=list(range(0, 360, 45)),
                ticktext=['N 0°','45°','E 90°','135°','S 180°','225°','W 270°','315°'],
                tickfont=dict(size=9)
            ),
            radialaxis=dict(
                range=[0, 90],
                tickvals=[15, 30, 45, 60, 75],
                ticktext=['75°', '60°', '45°', '30°', '15°'],  # displayed as elevation
                tickfont=dict(size=9),
                gridcolor='rgba(0,0,0,0.15)'
            )
        )

        # ════════════════════════════════════════════════════════════════════════
        # ROW 4 — Ground track map  [ORIGINAL — unchanged]
        # ════════════════════════════════════════════════════════════════════════
        fig.add_trace(
            go.Scattergeo(
                lon=longitudes, lat=latitudes,
                mode='markers+lines',
                marker=dict(size=3, color='black'),
                line=dict(width=1),
                name="Ground Track",
                showlegend=True
            ),
            row=4, col=1
        )

        fig.add_trace(
            go.Scattergeo(
                lon=[self.gs_data['longitude_deg']],
                lat=[self.gs_data['latitude_deg']],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='star'),
                text=[selected_gs],
                textposition="top right",
                name=selected_gs,
                showlegend=True
            ),
            row=4, col=1
        )

        for idx, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
            fig.add_trace(
                go.Scattergeo(
                    lon=[longitudes[start_idx]], lat=[latitudes[start_idx]],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='circle'),
                    name="Downlink Start" if idx == 0 else None,
                    showlegend=(idx == 0)
                ),
                row=4, col=1
            )
            fig.add_trace(
                go.Scattergeo(
                    lon=[longitudes[end_idx]], lat=[latitudes[end_idx]],
                    mode='markers',
                    marker=dict(size=10, color='orange', symbol='circle'),
                    name="Downlink End" if idx == 0 else None,
                    showlegend=(idx == 0)
                ),
                row=4, col=1
            )

        # preserve original geo range logic (last loop values of start_idx / end_idx)
        fig.update_geos(
            projection_type="natural earth",
            showcountries=True, showcoastlines=True, showland=True,
            landcolor="lightgray",
            lonaxis_range=[longitudes[start_idx] - 30, longitudes[end_idx] + 50],
            lataxis_range=[latitudes[start_idx] - 30, latitudes[end_idx] + 50],
            row=4, col=1
        )

        fig.update_layout(
            height=1500,
            title_text=(f"Ground Station Visibility and Satellite Ground Track<br>"
                        f"GS: {selected_gs} | Sat: {selected_satellite}"),
            title_x=0.5,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        html = fig.to_html(include_plotlyjs='cdn')
        self.ogs_plot.setHtml(html)
        # ------Save figure to file for browser viewing------
        output_dir = os.path.join(self.outputdir, 'figures')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'visibility_from_{selected_gs}.html')
        with open(output_file, 'w') as f:
            f.write(html)
        print(f"Full-resolution figure saved successfully to {output_file}")
    
    def compute_gimbal_angles(self, selected_gs, vis_times,
                          r_teme_m, v_teme_m_s):
        """
        Compute gimbal azimuth/elevation (LVLH frame)
        for satellite pointing to ground station.

        Inputs:
            selected_gs   : ground station name (str)
            vis_times     : astropy Time array (N,)
            r_teme_m      : satellite position (N,3) [meters, TEME]
            v_teme_m_s    : satellite velocity (N,3) [m/s, TEME]

        Returns:
            gimbal_az (deg), gimbal_el (deg)
        """
        #import numpy as np
        #import astropy.units as u
        #from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential

        # =========================
        # 1. Convert SAT → GCRS
        # =========================
        pos = CartesianRepresentation(
            r_teme_m[:, 0]*u.m,
            r_teme_m[:, 1]*u.m,
            r_teme_m[:, 2]*u.m
        )
        vel = CartesianDifferential(
            v_teme_m_s[:, 0]*u.m/u.s,
            v_teme_m_s[:, 1]*u.m/u.s,
            v_teme_m_s[:, 2]*u.m/u.s
        )
        # pos = CartesianRepresentation(r_teme_km * u.km)*1e3  # convert to meters
        # vel = CartesianDifferential(v_teme_km_s * u.km / u.s)*1e3  # convert to m/s
        teme = TEME(pos.with_differentials(vel), obstime=vis_times)
        gcrs = teme.transform_to(GCRS(obstime=vis_times))

        r_sat = gcrs.cartesian.xyz.to_value(u.m).T   # (N,3)
        v_sat = gcrs.velocity.d_xyz.to_value(u.m/u.s).T

        # =========================
        # 2. Ground station in GCRS
        # =========================
        r_gs, _ = self.ground_station_eci(selected_gs, start_time=self.datetime_edit.dateTime(), times = vis_times)  # should return (N,3) in meters

        # =========================
        # 3. LOS vector (sat → GS)
        # =========================
        rho = r_gs - r_sat
        rho_hat = rho / np.linalg.norm(rho, axis=1)[:, None]

        # =========================
        # 4. LVLH frame (local to satellite)
        # =========================
        r_norm = np.linalg.norm(r_sat, axis=1)[:, None]
        z_hat = -r_sat / r_norm                # along nadir

        h = np.cross(r_sat, v_sat)
        h_norm = np.linalg.norm(h, axis=1)[:, None]
        y_hat = -h / h_norm                    # opposite orbit normal

        x_hat = np.cross(y_hat, z_hat)         # completes right-handed LVLH

        # =========================
        # 5. Transform LOS → LVLH
        # =========================
        rho_lvlh = np.stack([
            np.sum(rho_hat * x_hat, axis=1),
            np.sum(rho_hat * y_hat, axis=1),
            np.sum(rho_hat * z_hat, axis=1)
        ], axis=1)

        x, y, z = rho_lvlh[:, 0], rho_lvlh[:, 1], rho_lvlh[:, 2]

        # =========================
        # 6. Compute gimbal angles
        # =========================
        gimbal_az = np.degrees(np.arctan2(y, x))
        gimbal_el = np.degrees(np.arcsin(z))

        # unwrap azimuth to avoid jumps
        # gimbal_az = np.degrees(np.arctan2(y, x))
        gimbal_az = (gimbal_az + 180) % 360 - 180
        
        # gimbal_az = np.degrees(np.unwrap(np.radians(gimbal_az)))

        return gimbal_az, gimbal_el

    def downsample_data(self,data_dict, max_points=1000):
        """Downsample data if it exceeds max_points"""
        total_points = len(data_dict['time'])
        
        if total_points > max_points:
            idx = np.linspace(0, total_points - 1, max_points).astype(int)
            downsampled = {
                'time': data_dict['time'][idx],
                'pe': data_dict['pe'][idx],
                'quat_error': data_dict['quat_error'][idx],
                'q_true': data_dict['q_true'][idx, :],
                'q_pred': data_dict['q_pred'][idx, :],
            }
            # Optional keys for row 3
            if 't_from_0' in data_dict:
                downsampled['t_from_0'] = data_dict['t_from_0'][idx]
            if 't_stamps_updates' in data_dict:
                downsampled['t_stamps_updates'] = data_dict['t_stamps_updates']
            if 'data_full_att_h' in data_dict:
                downsampled['data_full_att_h'] = data_dict['data_full_att_h']
            
            note = f" (showing {max_points} of {total_points} points)"
        else:
            downsampled = data_dict.copy()
            note = ""
        
        return downsampled, note

    def update_pe_visualization(self):
        """Update the pointing error visualization with optional swapped quaternion comparison"""
        if not self.pe_data:
            return
        
        # Check if we have swapped data
        has_swapped = hasattr(self, 'pe_data_swapped') and self.pe_data_swapped is not None
        # Downsample both datasets
        pe_data_display, downsample_note = self.downsample_data(self.pe_data)
        if has_swapped:
            pe_data_swapped_display, _ = self.downsample_data(self.pe_data_swapped)
        
        # Check if we have all required keys for 3-row plot
        required_keys = ['t_from_0', 'q_true', 'q_pred', 't_stamps_updates', 'data_full_att_h']
        add_qc_comparison = all(k in self.pe_data and self.pe_data[k] is not None for k in required_keys)
        #add_qc_comparison = False
        # Determine subplot configuration
        if add_qc_comparison:
            num_rows = 4
            row_heights = [1.0/num_rows] * num_rows # [0.25,0.25,0.25,0.25]
            subplot_titles = ('Pointing Error', 'Quaternion Error', 'Quaternion Components', 'QC Comparison')
            height = 1200
        else:
            num_rows = 3
            row_heights = [1.0/num_rows] * num_rows #[0.33,0.33,0.33]
            subplot_titles = ('Pointing Error', 'Quaternion Error', 'Quaternion Components')
            height = 800
        
        # Create figure
        fig = make_subplots(
            rows=num_rows, cols=1,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
            vertical_spacing=0.08,
            row_heights=row_heights
        )
        
        # Helper function to add traces for a dataset
        def add_pe_traces(data, prefix='', line_style=None, row=1):
            """Add pointing error traces"""
            trace_config = {'name': f'{prefix}PE'.strip()}
            if line_style:
                trace_config['line'] = line_style
            
            fig.add_trace(
                go.Scatter(x=data['time'], y=data['pe'], **trace_config),
                row=row, col=1
            )

        def add_quat_error_traces(data, prefix='', line_style=None, row=2):
            """Add Quaternion error traces"""
            trace_config = {'name': f'{prefix}QE'.strip()}
            if line_style:
                trace_config['line'] = line_style
            
            fig.add_trace(
                go.Scatter(x=data['time'], y=data['quat_error'], **trace_config),
                row=row, col=1
            )
        
        def add_quaternion_traces(data, prefix='', dash_style=None, row=3):
            """Add quaternion component traces"""
            for i, comp in enumerate(['w', 'x', 'y', 'z']):
                # True quaternion
                true_config = {'name': f'{prefix}True q{comp}'.strip()}
                fig.add_trace(
                    go.Scatter(x=data['time'], y=data['q_true'][:,i], **true_config),
                    row=row, col=1
                )
                
                # Predicted quaternion
                pred_config = {'name': f'{prefix}Pred q{comp}'.strip(), 'line': {'dash': dash_style or 'dash'}}
                fig.add_trace(
                    go.Scatter(x=data['time'], y=data['q_pred'][:,i], **pred_config),
                    row=row, col=1
                )
        
        def add_qc_comparison_traces(data, prefix='', marker_symbol='circle', row=4):
            """Add QC comparison traces (matplotlib-style)"""
            ii_q = 0  # Component index to plot
            t_from_0 = data.get('t_from_0')
            q_host_true_5ms = data.get('q_true')
            q_host_pred_5ms = data.get('q_pred')
            t_stamps_updates = data.get('t_stamps_updates')
            data_full_att_h = data.get('data_full_att_h')
            
            # Only plot update markers if arrays are valid
            if (t_stamps_updates is not None and
                data_full_att_h is not None and
                len(t_stamps_updates) > 0 and
                len(data_full_att_h.shape) == 2 and
                data_full_att_h.shape[0] == len(t_stamps_updates)):
                
                fig.add_trace(
                    go.Scatter(
                        x=t_stamps_updates, 
                        y=data_full_att_h[:, ii_q], 
                        mode='markers', 
                        marker=dict(size=10, symbol=marker_symbol), 
                        name=f'{prefix}Updates'.strip()
                    ),
                    row=row, col=1
                )
            
            # True trajectory
            fig.add_trace(
                go.Scatter(
                    x=t_from_0, 
                    y=q_host_true_5ms[:, ii_q], 
                    mode='lines', 
                    name=f'{prefix}True'.strip()
                ),
                row=row, col=1
            )
            
            # Predicted trajectory
            fig.add_trace(
                go.Scatter(
                    x=t_from_0, 
                    y=q_host_pred_5ms[:, ii_q], 
                    mode='lines+markers', 
                    marker=dict(size=2, symbol=marker_symbol), 
                    name=f'{prefix}Predicted'.strip()
                ),
                row=row, col=1
            )
        
        # Row 1: Pointing Error
        add_pe_traces(pe_data_display, prefix='')
        if has_swapped:
            add_pe_traces(pe_data_swapped_display, prefix='Swapped ', line_style={'dash': 'dot'})
        
        add_quat_error_traces(pe_data_display, prefix='')
        if has_swapped:
            add_quat_error_traces(pe_data_swapped_display, prefix='Swapped ', line_style={'dash': 'dot'})
        
        # Row 2: Quaternion Components
        add_quaternion_traces(pe_data_display, prefix='')
        if has_swapped:
            add_quaternion_traces(pe_data_swapped_display, prefix='Swapped ', dash_style='dot')
        
        # Row 3: QC Comparison (if data available)
        if add_qc_comparison:
            add_qc_comparison_traces(pe_data_display, prefix='')
            if has_swapped:
                add_qc_comparison_traces(pe_data_swapped_display, prefix='Swapped ', marker_symbol='x')
        
        # Update layout and axes
        title_suffix = ' (with Swapped Comparison)' if has_swapped else ''
        fig.update_layout(
            height=height, 
            showlegend=True,
            title_text=f"Pointing Error Visualization{title_suffix}{downsample_note}",
            title_x=0.5
        )
        
        # Y-axis labels
        fig.update_yaxes(title_text="PE [urad]", row=1, col=1)
        fig.update_yaxes(title_text="QE [urad]", row=2, col=1)
        fig.update_yaxes(title_text="Quaternion", row=3, col=1)
        if add_qc_comparison:            
            fig.update_yaxes(title_text="QC [-]", row=3, col=1)
        
        # X-axis labels
        fig.update_xaxes(title_text="Time [s]", row=num_rows-1 if num_rows == 3 else 3, col=1)
        if add_qc_comparison:
            fig.update_xaxes(title_text="t [s]", row=3, col=1)
        
        # Convert to HTML and display in GUI
        html = fig.to_html(include_plotlyjs='cdn')
        self.pe_plot.setHtml(html)
        
        plot_type = '3-row' if add_qc_comparison else '2-row'
        swap_msg = ' with swapped comparison' if has_swapped else ''
        print(f"PE Plotly HTML ({plot_type}{swap_msg}) set for pe_plot.")

        ##-----------NEW ANIMATION 12/06/26-------------------------
        import pyvista as pv
        # =====================================================
        # CUBES
        # =====================================================
        plotter = pv.Plotter(window_size=(1400, 900))

        cube_true = pv.Cube(
            center=(0,0,0),
            x_length=1,
            y_length=1,
            z_length=1
        )

        cube_est = pv.Cube(
            center=(0,0,0),
            x_length=1,
            y_length=1,
            z_length=1
        )

        
        actor_true = plotter.add_mesh(
            cube_true,
            color="limegreen",
            opacity=0.6,
            label="True"
        )

        actor_est = plotter.add_mesh(
            cube_est,
            color="red",
            opacity=0.6,
            label="Estimated"
        )

        plotter.add_axes()
        # Labels
        plotter.add_point_labels(
            np.array([[-3, 0, 1.2]]),
            ["TRUE"],
            font_size=20,
            point_size=0
        )

        plotter.add_point_labels(
            np.array([[3, 0, 1.2]]),
            ["ESTIMATED"],
            font_size=20,
            point_size=0
        )

        text_actor = plotter.add_text(
            "",
            position="upper_left",
            font_size=12
        )
        plotter.camera_position = [
            (0, -10, 6),   # camera
            (0, 0, 0),     # focal point
            (0, 0, 1)      # up direction
        ]


        # plotter.show(auto_close=False)
        # =====================================================
        # ANIMATION LOOP
        # =====================================================
        print("First true quat:", pe_data_display['q_true'][0])
        print("Last true quat :", pe_data_display['q_true'][-1])

        print("First est quat :", pe_data_display['q_pred'][0])
        print("Last est quat  :", pe_data_display['q_pred'][-1])

        # k = 0
        # def update():
        #     global k

        #     if k >= len(pe_data_display['q_true']):
        #         return

        #     Rt = self.accessibleNamequat_to_rotmat(pe_data_display['q_true'][k])
        #     Re = self.accessibleNamequat_to_rotmat(pe_data_display['q_pred'][k])

        #     actor_true.SetUserMatrix(self.make_transform(Rt, [-3,0,0]))
        #     actor_est.SetUserMatrix (self.make_transform(Re, [ 3,0,0]))

        #     k += 1
        # plotter.add_callback(update, interval=30)
        # plotter.show()
        # for k in range(len(pe_data_display['q_true'])):
        #     if k % 100 == 0:
        #         print("Frame", k)

        #     Rt = self.quat_to_matrix(pe_data_display['q_true'][k])
        #     Re = self.quat_to_matrix(pe_data_display['q_pred'][k])

        #     if k == 0:
        #         print(np.max(np.abs(pe_data_display['q_true'] - pe_data_display['q_pred'])))

        #     if k == len(pe_data_display['q_true'])-1:
        #         print(Rt)

        #     # actor_true.user_matrix = self.make_transform(Rt,[-3, 0, 0])
        #     # actor_est.user_matrix = self.make_transform(Re,[ 3, 0, 0])

        #     actor_true.SetPosition(-3, 0, 0)
        #     actor_est.SetPosition( 3, 0, 0)

        #     actor_true.SetOrientation(
        #         *R.from_matrix(Rt).as_euler('xyz', degrees=True)
        #     )

        #     actor_est.SetOrientation(
        #         *R.from_matrix(Re).as_euler('xyz', degrees=True)
        #     )

        #     # text_actor.SetInput(
        #     #     f"Quaternion Error: {pe_data_display['quat_error'][k]:.3f} deg"
        #     # )
        #     plotter.remove_actor(text_actor)

        #     text_actor = plotter.add_text(
        #         f"Frame: {k}\n"
        #         f"Orientation Error: {pe_data_display['quat_error'][k]:.3f} deg",
        #         position="upper_left",
        #         font_size=14
        #     )

        #     # plotter.render()
        #     # adjust speed if needed
        #     time.sleep(0.02)

        # plotter.close()

        ##-----------END NEW ANIMATION------------------------------

    def eci_to_azel(self, sat_pos_eci_km, obs_time_utc, gs_lat_deg, gs_lon_deg, gs_alt_m, orientation='ENU'):
        """
        Convert satellite ECI position to Azimuth and Elevation angles
        as seen from a ground station.

        Parameters:
        - sat_pos_eci_km: satellite position vector in ECI (GCRS) frame [x, y, z] in km
        - obs_time_utc: observation time as ISO string or astropy Time object in UTC
        - gs_lat_deg, gs_lon_deg: ground station latitude and longitude in degrees
        - gs_alt_m: ground station altitude in meters
        - orientation: 'ENU' Default orientation for Az/El in Astropy 

        Returns:
        - azimuth_rad: azimuth angle in rad (0°=North, clockwise)
        - elevation_rad: elevation angle in rad (0°=horizon, 90°=zenith)
        """
        # Ensure obs_time is an astropy Time object in UTC scale
        if not isinstance(obs_time_utc, Time):
            obs_time_utc = Time(obs_time_utc, scale='utc')

        # Define ground station location
        gs_location = EarthLocation(lat=gs_lat_deg*u.deg, lon=gs_lon_deg*u.deg, height=gs_alt_m*u.m)

        # Convert satellite ECI position (km) to Cartesian representation astropy object
        sat_pos_eci = CartesianRepresentation(np.array(sat_pos_eci_km) * u.km)
        
        # Create GCRS coordinate of satellite at obs_time
        sat_gcrs = GCRS(sat_pos_eci, obstime=obs_time_utc)
        
        # Convert satellite position from ITRS to AltAz (ENU relative to ground station)
        altaz_frame = AltAz(location=gs_location, obstime=obs_time_utc)
        sat_altaz = sat_gcrs.transform_to(altaz_frame)

        # Astropy ENU Az/El
        azimuth_rad     = sat_altaz.az.to(u.rad).value
        elevation_rad   = sat_altaz.alt.to(u.rad).value

        if orientation.upper() == "ENU":
            pass
        
        elif orientation.upper() == "NWU":
            azimuth_rad = (-azimuth_rad ) % (2*np.pi)
            
        elif orientation.upper() == "NED":
            elevation_rad = -elevation_rad
        else:
            raise ValueError("orientation must be ENU, NWU, or NED")

        return azimuth_rad, elevation_rad
       
    def ground_station_eci(self, selected_gs, start_time, times=None):
        """
        Returns position and velocity of ground station in GCRS (ECI)
        """
        self.gs_data = self.ground_stations_data.get(selected_gs)
        if not self.gs_data:
            QMessageBox.warning(self, "Warning", f"No data available for ground station {selected_gs}")
            return

        if times is None:
            times = Time(start_time.toPyDateTime().replace(tzinfo=timezone.utc)) + np.arange(0, 3600, 5)*u.s  # 1 hour of data at 5 sec intervals
        # Convert QDateTime -> Python datetime with UTC
        #start_dt_utc = start_time.toPyDateTime().replace(tzinfo=timezone.utc)

        # Create Astropy Time array
        #times = Time(start_dt_utc) + np.arange(0, 3600, 5)*u.s  # 1 hour at 5 sec steps


        # Extract ground station coordinates
        gs_location = EarthLocation(lat =self.gs_data['latitude_deg']* u.deg, 
                                    lon   =self.gs_data['longitude_deg']* u.deg, 
                                    height=self.gs_data['altitude_km']* u.km)
        
        
        gs_gcrs = gs_location.get_gcrs(obstime=times)

        r_eci = gs_gcrs.cartesian.xyz.to_value(u.m)
        v_eci = gs_gcrs.velocity.d_xyz.to_value(u.m/u.s)
        t_gps = times.gps
       
        # Build DataFrame
        df = pd.DataFrame({
            't_gps': t_gps,
            'r_x':   r_eci[0, :],
            'r_y':   r_eci[1, :],
            'r_z':   r_eci[2, :],
            'v_x':   v_eci[0, :],
            'v_y':   v_eci[1, :],
            'v_z':   v_eci[2, :],
        })

        # Save to CSV
        filename = os.path.join(self.outputdir, 'tables','GS', f'{selected_gs}_ECI.csv')
        df.to_csv(filename, index=False, float_format='%.6f')

        print(f"""Saved {len(df)} epochs to {filename}
                Time range: {times[0].iso} → {times[-1].iso}
                Ground Station: {selected_gs}
                GPS time range: {t_gps[0]:.1f} → {t_gps[-1]:.1f} s""")

        
        return r_eci.T, v_eci.T

    def eci_to_body_quaternion(self, lat_deg, lon_deg, orientation, time):
        gs = self.ground_station_location(lat_deg, lon_deg, 0.0)

        # GCRS basis vectors in ECEF
        itrs = gs.get_itrs(obstime=time)
        dcm_ecef2eci = itrs.transform_to(GCRS(obstime=time)).cartesian.xyz.value
        dcm_eci2ecef = dcm_ecef2eci.T

        dcm_ecef2ned = dcm_ecef_to_ned(lat_deg, lon_deg)
        dcm_ned2local = dcm_ned_to_local(orientation)

        dcm_eci2body = dcm_ned2local @ dcm_ecef2ned @ dcm_eci2ecef

        quat = Rotation.from_matrix(dcm_eci2body).as_quat()  # [x,y,z,w]

        return quat

    def sun_eci_full(self):
        from pytz import timezone
        """
        outputs:
            inputs= t_gps
                    r_gs_eci                # r, v [m, m/s] of HQ in ECI  
                    r_target_eci            # r, v [m, m/s]. v = 0 for sun
                    quat_eci2bf_returned    # scalar-first quat from ECI to NED
                    quat_bf2gf_returned]    # scalar-first quat from NED to Terminal Global Frame (North West Up)

            ae_2_sun                        # Az, El [rad]

        """
        tud_rotator = tudatconv.tudat_predictor()
        mynaric_TBRW = [48.1372, 11.4193828, 567.5] # Mynaric postion coordinates
        ogs_location = self.ogs_geo_location

        selected_body = self.celes_item.currentText()     
        if selected_body == 'Sun':   
            body = 'sun'
        elif selected_body == 'Moon':
            body = 'moon'

        t_start_cest = dt.datetime.now()        
        output_dir = os.path.join(self.outputdir, 'tables', f'SUN_MOON')
        os.makedirs(output_dir, exist_ok=True)        
        full_output_folder = f'{output_dir}//{selected_body}_{t_start_cest.date().isoformat()}'
        zip_name = f'{body}_{t_start_cest.date().isoformat()}.zip'

        try:            
            os.mkdir(full_output_folder)
            print(f'Made folder {full_output_folder}')
        except:
            pass      

        inputs, ae_now = ae_roof2sun(t_start_cest, 
                                     tud_rotator, 
                                     mode = body, 
                                     coord_hq = ogs_location, #mynaric_TBRW, 
                                     pointing = 'NWU',
                                     force_unity_quaternion = 1)
        
        msg = f"AZ/El[deg]: {ae_now}\
            \nGPS_time: {np.round(inputs[0])}\
            \nGround states[ECI] : {inputs[1]}\
            \nSun Ephemeris[ECI]: {inputs[2]}\
            \nQ_ECI2BF: {inputs[3]}\
            \nq_bf2gf: {inputs[4]}"
        
        t_start_utc = Time(t_start_cest, scale='utc')
        # self.generate_gs_eci_data(
        #     gs_coord_wgs84=mynaric_TBRW, 
        #     t_gps_0=t_start_utc.gps,  # GPS time in seconds
        #     dt_step=1,
        #     t_prop=7200,  # 1 hours propagation
        #     orientation="nwu",  # north west up
        #     output_path= full_output_folder
        # )  # save folder

        ###-------------------------- New method---------------------------
       
        # Example usage:

        # Satellite position in ECI frame (GCRS) in kilometers (example values for ISS)
        satellite_eci_km = inputs[2]*1e-3# [6524.834, 6862.875, 6448.296]  # example ECI position vector (km)

        # Observation time (UTC)
        
        obs_time = Time(inputs[0], format='gps').utc # Convert to Astropy Time object
        # Ground station location (Munich approx)
        gs_latitude = mynaric_TBRW[0]   # degrees
        gs_longitude = mynaric_TBRW[1]  # degrees
        gs_altitude = mynaric_TBRW[2]   # meters

        azimuth, elevation = self.eci_to_azel(satellite_eci_km, obs_time, gs_latitude, gs_longitude, gs_altitude)

        print(f"ECI psotion input in Km: {satellite_eci_km}")
        print(f"Azimuth:\tASTROPY: {azimuth:.2f}°\t ASTRAA:{360-ae_now[0]}°")
        print(f"Elevation:\tASTROPY: {elevation:.2f}°\t ASTRAA:{ae_now[1]}°")
        print(f"Time:\tASTROPY: {obs_time.iso}UTC {inputs[0]}GPS\t ASTRAA:{inputs[0]}GPS")

        ###-------------------------- END New method---------------------------
        # Show the message of instantaneous values
        QMessageBox.information(self, "Values at Current time", msg)

        ##------------------Cretating csv files for all data--------------

        start_date_q = self.start_date_edit.date()
        end_date_q = self.end_date_edit.date()

        start_date_dt = start_date_q.toPyDate()
        end_date_dt = end_date_q.toPyDate()
        t_start_cest = dt.datetime(start_date_dt.year, start_date_dt.month, start_date_dt.day)

        # loop_time = t_start_cest
        # sun_eci_pos_vel =[]
        # pbar = tqdm(desc="Processing", unit="%", ncols=100, ascii=True,dynamic_ncols=True)
        
        h_start = t_start_cest.hour 
        nr_days =  1
        h_end = 24
        t_start_local = t_start_cest#dt.datetime(currrent_year, month_used, day_used, h_start, min_start, 0) 
        # Save folder
        output_dir = os.path.join(self.outputdir, 'tables', f'SUN_MOON')
        os.makedirs(output_dir, exist_ok=True)        
        full_output_folder = f'{output_dir}//{body}_{t_start_cest.date().isoformat()}'
        zip_name = f'{body}_{t_start_cest.date().isoformat()}.zip'

        ## remove old folder
        try:
            import shutil
            shutil.rmtree(full_output_folder)
            print(f"Removed old folder: {full_output_folder}")
        except Exception as e:
            print(f"Error removing folder {full_output_folder}: {e}")

        try:            
            os.mkdir(full_output_folder)
            print(f'Made folder {full_output_folder}')
        except:
            pass

        dt_mins = 1 / 60 # 1 second timesteps
        # dt_mins = 1 / 60 / 200 # 5 milisecond timesteps
        # dt_mins = 1 / 60 / 2 # .5 second timesteps
        # dt_mins = 1 / 60 / 20 # .05 second steps -> ~4 urad 

        n_digits_used = len(str(dt_mins*60))-2 # 3 digits for 5 ms

        dt_loop = dt.timedelta(minutes = dt_mins)
        constant_dt_upd = dt.timedelta(seconds = 1)
        loop_length = int(60 / dt_mins* (h_end-h_start + (nr_days-1)*24))
        
        # placeholders
        cest_hoursmins = []
        t_gps       = np.zeros((loop_length, 1))
        states_gs   = np.zeros((loop_length, 6)) 
        states_sun  = np.zeros((loop_length, 6))
        q_eci2bf    = np.zeros((loop_length, 4))  
        q_bf2gf     = np.zeros((loop_length, 4))   
        ae_data     = np.zeros((loop_length, 2))   
        
        # get J2000 time
        month_used = start_date_dt.month
        if month_used > 3 and month_used < 11:
            dt_local2utc = -2
        else:
            dt_local2utc = -1
        dt_gps2j2000 = t_conv.dt_gps2j2000tt() # t_j2000 = t_gps + dt_
        t_j2000_start = t_conv.utc2gws(t_start_local+ dt.timedelta(hours = dt_local2utc)) + dt_gps2j2000
        datetime_update_rate = int(1/dt_mins/60)


        # for ii in range(loop_length):
        for ii in tqdm(range(loop_length), desc="Processing", unit="%", ncols=100, ascii=True):
            if ii == 0:
                t_current = t_start_local
                t_j2000_current = t_j2000_start
            inputs, ae_2_sun = ae_roof2sun(t_current, 
                                        tud_rotator, 
                                        mode = body,
                                        pointing="NWU",
                                        manual_az_correction = 0, 
                                        mounting_offset = 0, 
                                        t_j2000 = t_j2000_current, dt_gps2j2000 = dt_gps2j2000, 
                                        force_unity_quaternion = 1)

            cest_hoursmins.append(t_current.time().isoformat())
            ae_data[ii,:] = np.deg2rad(ae_2_sun) # rad
            t_gps[ii,:] = np.round(inputs[0], n_digits_used)
            states_gs[ii,:] = inputs[1]
            states_sun[ii,:3] = inputs[2]
            q_eci2bf[ii, :4] = inputs[3]
            q_bf2gf[ii, :4] = inputs[4]
            t_j2000_current  = t_j2000_current + dt_mins*60
            
            if ii % datetime_update_rate == 0 and ii >= datetime_update_rate:
                # Update CEST date-time tracker every 1 second
                t_current = t_current + constant_dt_upd

        cest_hoursmins = np.array(cest_hoursmins).reshape((loop_length, 1))

        title_reftime = 'ref_time'
        output_times_dict = { 'year_used' : [start_date_dt.year],
            'month_used' : [start_date_dt.month],
            'day_used' : [start_date_dt.day],
            'h_start' : [h_start],
            'h_end' : [h_end],
            't_res' : [dt_mins*60]}
        df_date = pd.DataFrame.from_dict(output_times_dict)
        df_date.to_csv(f"{full_output_folder}//{title_reftime}.csv", index = False)
        print(f'Loop done. Start : {t_start_local} \nFinal time : {t_current}')

        
        ##get EA
        ea_all = np.zeros((q_eci2bf.shape[0],3))
        omega_all = np.zeros((q_eci2bf.shape[0],3))
        for ii, q_ii in enumerate(q_eci2bf):
            ea_ii = att_conv.convert_dcm2ea(att_conv.convert_quat2dcm(q_ii)) # deg
            ea_all[ii,:] = ea_ii

        ## get EA dot
        ea_dot = np.zeros(ea_all.shape)    
        for ii in range(3):
            ea_dot[:,ii] = np.gradient(ea_all[:,ii], t_gps.flatten())
            
        ## get Omega
        ## get q; qdot
        q_full = np.zeros((ea_all.shape[0],8))
        
        for ii, ea_ii in enumerate(ea_all):
            omega_ii = att_conv.calc_omega(ea_ii, ea_dot[ii,:], deg = 1) # deg/s            
            q_recal, q_dot = att_conv.calc_qdot(ea_ii, omega_ii)
            q_full[ii,:4] = q_recal.flatten()
            q_full[ii,4:] = q_dot.flatten()
        q_eci2bf = q_full

        output_success, df = io.save_azel(t_gps,
                                        states_gs,
                                        states_sun,
                                        q_eci2bf,
                                        q_bf2gf,
                                        ae_data,
                                        fname = f'gs2{selected_body}_data',
                                        full_folder = full_output_folder,
                                        zip_name = zip_name,
                                        make_zip = 1)
        #-----------------Plot Sun Az/El------------------------------------
        
        utc_naive = Time(t_gps,format='gps').utc.to_datetime().flatten()
        #local_tz = timezone('Europe/Berlin')
        times_for_plot = [dt.replace(tzinfo=timezone('UTC')).astimezone(CET) for dt in utc_naive]

        az_for_plot  = np.rad2deg(ae_data[:,0])
        alt_for_plot = np.rad2deg(ae_data[:,1])

        filename = f"gs2{selected_body}"
        output_path = os.path.join(full_output_folder, f"{filename}.svg")

        fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        ax1.plot(times_for_plot, alt_for_plot, label = 'Elevation')
        ax2.plot(times_for_plot, az_for_plot, label = 'Azimuth')
        ax2.set_xlabel("Local Time")
        ax1.set_ylabel("Elevation[°]")
        ax2.set_ylabel("Azimuth [°]")        
        ax1.grid(True)
        ax2.grid(True)
        ax1.legend()
        ax2.legend()
        plt.title(f'Ground station to {selected_body}')
        plt.tight_layout()
        plt.show()

        plt.savefig(output_path)

        fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                            subplot_titles=(f'{selected_body} Altitude/Azimuth'))

        fig.add_trace(go.Scatter(x=times_for_plot, y=az_for_plot, mode='lines+markers', name='Azimuth',marker=dict(size=4), line=dict(color='green')))
        fig.add_trace(go.Scatter(x=times_for_plot, y=alt_for_plot, mode='lines+markers', name='Elevation',marker=dict(size=4), line=dict(color='purple')))

        fig.update_layout(
            title='Plot of Local Times',
            xaxis_title='Local Time',
            yaxis_title='Value',
            height=600,
            hovermode='x unified'
        )
        
        fig.write_html(os.path.join(full_output_folder,f"{filename}.html"))   
        

        QMessageBox.information(self, "Full data saved",full_output_folder)
  
    def plot_sky(self, df, az_col='az_rad', el_col='el_rad', title="Sky Plot", mode='lines+markers'):
        """
        df: DataFrame with azimuth and elevation in radians
        az_col: column name for azimuth
        el_col: column name for elevation
        mode: 'lines', 'markers', or 'lines+markers'
        """
        # Remove NaNs
        df_clean = df.dropna(subset=[az_col, el_col])
        
        # Convert elevation to radial distance: center=90°, outer=0°
        r = 90 - np.rad2deg(df_clean[el_col].to_numpy())
        r = np.clip(r, 0, 90)  # ensure within 0-90

        # Azimuth in degrees, 0° top, clockwise
        theta = (90 - np.rad2deg(df_clean[az_col].to_numpy())) % 360

        fig = go.Figure(go.Scatterpolar(
            r=r,
            theta=theta,
            mode=mode,
            line=dict(color='blue'),
            marker=dict(size=6, color='red'),
            name='Trajectory'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 90],
                    tickvals=[0, 30, 60, 90],
                    ticktext=['90°', '60°', '30°', '0°'],
                    showline=True
                ),
                angularaxis=dict(
                    direction='clockwise',
                    rotation=90,
                    tickmode='array',
                    tickvals=[0, 90, 180, 270],
                    ticktext=['0°','90°','180°','270°']
                )
            ),
            title=title,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        return fig

    def getfile(self):      
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  
        self.data_fname,_ = QFileDialog.getOpenFileNames(self, 'Open TLE file', '',"files (*.txt)", options = options)
        print("\nCustom TLE file loaded...")
    
    def telescope_ecef_quat(self):
                
        start_time = self.datetime_edit.dateTime().toPyDateTime().replace(tzinfo=timezone.utc)
        duration_sec  = 720 # seconds
        dt = 1.0  
        
        # --- Construct list of times ---
        num_steps = int(duration_sec / dt) + 1  # include last step
        times = [start_time + timedelta(seconds=i*dt) for i in range(num_steps)]
        
        # Get selected ground station
        selected_gs = self.gs_selector.currentText()
        if not selected_gs or selected_gs == "No Ground Stations Found":
            QMessageBox.warning(self, "Warning", "Please select a ground station first.")
            return

        # Get ground station data
        self.gs_data = self.ground_stations_data.get(selected_gs)
        if not self.gs_data:
            QMessageBox.warning(self, "Warning", f"No data available for ground station {selected_gs}")
            return
        
        # WGS84" Ellipsoide
        gs_ecef = EarthLocation.from_geodetic(lat   = self.gs_data['latitude_deg'], 
                                            lon     = self.gs_data['longitude_deg'], 
                                            height  = self.gs_data['altitude_km']* 1000.0 * u.m)

        target_type = self.OGS_combo.currentIndex()
        satellite_name = self.sat_selector.currentText()
        
        resultrows = self.track_target(times, site_ecef = gs_ecef, target_type=target_type, sat=None, frame=self.frame_combo.currentText())        
        df = pd.DataFrame(resultrows)
        
        # Save to CSV
        if target_type == 1:
            target_name = "Sun"
        elif target_type == 2:
            target_name = "Moon"
        elif target_type == 3:
            target_name = "CustomSat"
        else:
            target_name = satellite_name if satellite_name else "Satellite"
        csv_filename = os.path.join(self.outputdir,'track_log', f"{target_name}_tracking.csv")
        
        with open(csv_filename, "w") as f:
            f.write(f"# {target_name}, {start_time.isoformat()},  {self.frame_combo.currentText()}-Frame\n")
            df.to_csv(f, index=False, float_format="%.9f")  
        print(f"Saved 10 mins.tracking data to {csv_filename}") 

        # Plot Figure
        time_for_plot = Time(df['gps_time'].to_numpy(), format='gps', scale='utc').to_datetime()
        plt.figure(figsize=(10,6))
        plt.plot(time_for_plot, df['el_rad']*180/np.pi, label="Elevation (deg)")
        plt.plot(time_for_plot, df['az_rad']*180/np.pi, label="Azimuth (deg)")
        plt.xlabel("UTC Time")
        plt.ylabel("Angle (deg)")
        plt.title(f"{target_name} Tracking Angles from {selected_gs}, {self.frame_combo.currentText()}-Frame")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        output_filename = os.path.join(self.outputdir,'track_log',f"{target_name}_pass_from {selected_gs}, {self.frame_combo.currentText()}-Frame.png")
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {output_filename}")


        fig = self.plot_sky(df, az_col='az_rad', el_col='el_rad', title=f"{target_name} Tracking Angles from {selected_gs}")
        html = fig.to_html(include_plotlyjs='cdn')
        self.ogs_plot.setHtml(html)

    def track_target(self, times, site_ecef, target_type=0, sat=None, frame="NWU"):
        """
        Generic tracker for satellite or Sun.

        Parameters
        ----------
        times : list of datetime.datetime
            Times in UTC.
        site_ecef : array_like
            Observer ECEF position [x, y, z] in meters.
        lat_deg, lon_deg : float
            Observer latitude and longitude in degrees.
        target_type : str
            'satellite' or 'sun'.
        sat : sgp4.Satellite (optional)
            Required if target_type='satellite'.
        frame : str
            Local frame for telescope orientation: 'NWU', 'ENU', 'NED'.
        t_conv : object
            Object with `sky_utc2gps` method to convert datetime -> GPS time.

        Returns
        -------
        rows : list of dict
            Each row has keys: gps_time, az_rad, el_rad, q0..q3, wx_rad_s..wz_rad_s, qdot0..qdot3
        """
        rows = []
        q_prev = None
        t_prev = None

        # site = np.array([site_ecef['x'].to_value(u.m),
        #                 site_ecef['y'].to_value(u.m),
        #                 site_ecef['z'].to_value(u.m),
        #                 ])
        site = np.array([
            site_ecef.x.to_value(u.m),
            site_ecef.y.to_value(u.m),
            site_ecef.z.to_value(u.m)
            ])
        location = EarthLocation(
            x=site[0]*u.m,
            y=site[1]*u.m,
            z=site[2]*u.m
        )
        
        tle_line1 = None
        tle_line2 = None

        # target_type = self.OGS_combo.currentIndex()
        if target_type == 3:
            self.getfile()
            tle_file = self.data_fname[0]
            tle_line1, tle_line2 = self.read_single_tle(tle_file)

            if tle_line1 is None or tle_line2 is None:
                QMessageBox.warning(self, "Warning", "Failed to read custom TLE file.")
                return []
        
        elif target_type == 0:
            selected_satellite = self.sat_selector.currentText()
            if not selected_satellite:
                QMessageBox.warning(self, "Warning", "Please select a satellite first.")
                return

            # Get TLE data for the selected satellite
            tle_data = self.update_tle_info(selected_satellite)                    
            if not tle_data:
                QMessageBox.warning(self, "Warning", "No TLE data available for the selected satellite.")
                return

            # Get infotmation from the TLE
            tle_line1 = tle_data.get('line1')
            tle_line2 = tle_data.get('line2')
            if tle_line1 is None or tle_line2 is None:
                QMessageBox.warning(self, "Warning", "TLE data is incomplete (missing line1/line2).")
                return []
        
        if target_type in (0, 3) and (tle_line1 is None or tle_line2 is None):
            QMessageBox.warning(self, "Warning", "TLE lines are not set — cannot propagate.")
            return []
    
        if target_type in (0, 3):
            sat_propagator = Satrec.twoline2rv(tle_line1, tle_line2)

        for t in tqdm(times,total=len(times), desc=f"Processing Tracking data"):
            astropy_time = Time(t)

            # --- Target position ---
            if target_type == 1:
                altaz_frame = AltAz(obstime=astropy_time, location=location)
                sun_gcrs    = get_sun(astropy_time)
                sun_altaz   = sun_gcrs.transform_to(altaz_frame)
                az, el      = sun_altaz.az.rad, sun_altaz.alt.rad
                
            elif target_type == 2:    
                # tle_line1/2 already resolved above the loop            
                altaz_frame = AltAz(obstime=astropy_time, location=location)
                moon_gcrs   = get_moon(astropy_time, location=location)
                moon_altaz  = moon_gcrs.transform_to(altaz_frame)
                az, el      = moon_altaz.az.rad, moon_altaz.alt.rad
            
            elif target_type in (0, 3):
                #_,r_eci, v_eci = self.propagate_tle(tle_line1, tle_line2, astropy_time,output_frame="ECI")                
                # az, el = self.eci_to_azel(r_eci, astropy_time, 
                #                         gs_lat_deg  = self.gs_data['latitude_deg'], 
                #                         gs_lon_deg  = self.gs_data['longitude_deg'], 
                #                         gs_alt_m    = self.gs_data['altitude_km'],
                #                         orientation = frame)
    
                # ── FIX: bypass eci_to_azel entirely — use astropy GCRS→AltAz directly ──
                # propagate_tle returns r in GCRS (ECI), so wrap it as a GCRS frame object
                # and let astropy handle all frame rotations (GMST, polar motion, etc.)
                # sat_gcrs = GCRS(
                    # CartesianRepresentation(r_eci*u.km),
                    # obstime=astropy_time
                # )
                # altaz_frame = AltAz(obstime=astropy_time, location=location)
                # sat_altaz   = sat_gcrs.transform_to(altaz_frame)               
# 
                # az = float(np.pi/2-sat_altaz.az.rad)   # standard CW from North, [0, 2π) — same convention as Sun/Moon
                # el = float(sat_altaz.alt.rad)   # elevation above horizon, [−π/2, π/2]
                
                err, r_teme_km, v_teme_km_s = sat_propagator.sgp4(astropy_time.jd1, astropy_time.jd2)
                if err != 0:
                    print(f"[WARN] SGP4 error {err} at {astropy_time.iso}")

                pos = CartesianRepresentation(r_teme_km * u.km)
                vel = CartesianDifferential(v_teme_km_s * u.km / u.s)
                teme = TEME(pos.with_differentials(vel), obstime=t)
                #teme_pos  = TEME(CartesianRepresentation(r_teme[0]*u.km, r_teme[1]*u.km, r_teme[2]*u.km),obstime=astropy_time)
                gcrs_pos  = teme.transform_to(GCRS(obstime=astropy_time))
                #sat_altaz = gcrs_pos.transform_to(AltAz(obstime=astropy_time, location=location))
                
                r_eci = gcrs_pos.cartesian.xyz.to_value(u.km)
                v_eci = gcrs_pos.velocity.d_xyz.to_value(u.km/u.s)

                sat_altaz =teme.transform_to(AltAz(
                                        obstime=t,
                                        location=location,
                                    )
                                )
                az_geo = (sat_altaz.az.rad)    # CW from North [0, 2π) — matches STK, no offset needed   float((-sat_altaz.az.rad ) % (2*np.pi)) #
                az_ccw  = (2*np.pi - az_geo) % (2*np.pi)  # CCW from North [0, 2π) — 
                az = float(az_ccw)
                el = float(sat_altaz.alt.rad)   # above horizon

                    
            else:
                raise ValueError("target_type must be 'sun' or 'satellite'")

            # --- Telescope quaternion ---
            q = self.telescope_quaternion_from_azel(
                site_ecef=site,
                az=az,
                el=el,
                frame=frame
            )

            # --- Angular velocity & quaternion rate ---
            if q_prev is None:
                omega = np.zeros(3)
                qdot = np.zeros(4)
            else:
                dt_step = (t - t_prev).total_seconds()
                omega = quat_slerp.angular_velocity(q_prev, q, dt_step)
                qdot = quat_slerp.quaternion_derivative(q, omega)

            # --- Save row ---
            # gps_time = t_conv.sky_utc2gps(t) if t_conv else None
            gps_time = astropy_time.gps
            if target_type in (0, 3):                
                rows.append({
                    "gps_time"  : gps_time,
                    "az_rad"    : az,       "el_rad"    : el,
                    "q0"        : q[0],     "q1"        : q[1],     "q2"        : q[2],     "q3"    : q[3],
                    "qdot0"     : qdot[0],  "qdot1"     : qdot[1],  "qdot2"     : qdot[2],  "qdot3" : qdot[3],
                    "wx_rad_s"  : omega[0], "wy_rad_s"  : omega[1], "wz_rad_s"  : omega[2],
                    "T_pos_x"   : r_eci[0], "T_pos_y"   : r_eci[1],  "T_pos_z"  : r_eci[2],
                    "T_vel_x"   : v_eci[0], "T_vel_y"   : v_eci[1],  "T_vel_z"  : v_eci[2]

                })
            else:
                rows.append({
                    "gps_time"  : gps_time,
                    "az_rad"    : az,       "el_rad"    : el,
                    "q0"        : q[0],     "q1"        : q[1],     "q2"        : q[2],     "q3"    : q[3],
                    "qdot0"     : qdot[0],  "qdot1"     : qdot[1],  "qdot2"     : qdot[2],  "qdot3" : qdot[3],
                    "wx_rad_s"  : omega[0], "wy_rad_s"  : omega[1], "wz_rad_s"  : omega[2],

                })
            q_prev = q
            t_prev = t

        return rows
        
    def local_basis_from_ecef(self, site_ecef, frame="NWU"):
        """
        Returns local basis vectors (x, y, z) in ECEF
        according to requested frame convention.
        """
        up = self.normalize(site_ecef)

        z_hat = np.array([0.0, 0.0, 1.0])
        east = self.normalize(np.cross(z_hat, up))
        north = np.cross(up, east)

        frame = frame.upper()

        if frame == "NWU":
            x, y, z = north, -east, up            
        elif frame == "ENU":
            x, y, z = east,  north, up
        elif frame == "NED":
            x, y, z = north,  east, -up
        else:
            raise ValueError("frame must be 'NWU', 'ENU', or 'NED'")

        return x, y, z

    def telescope_quaternion_from_azel(self, site_ecef, az, el, frame="NWU"):
        """
        Compute ECEF → telescope quaternion (scalar-first)
        from az/el in a configurable local frame.
        """



        # Local basis in ECEF
        lx, ly, lz = self.local_basis_from_ecef(site_ecef, frame)
        frame = frame.upper()
        
        # NWU (lx=N, ly=W): E = -ly  → cos(az)*lx - sin(az)*ly
        # ENU (lx=E, ly=N): E =  lx  → sin(az)*lx + cos(az)*ly
        # NED (lx=N, ly=E, lz=Down): Up = -lz → cos(az)*lx + sin(az)*ly - sin(el)*lz

        if frame == "NWU":
            z_tel = (np.cos(el) * np.cos(az) * lx
                - np.cos(el) * np.sin(az) * ly
                + np.sin(el) * lz)
        elif frame == "ENU":
            z_tel = (np.cos(el) * np.sin(az) * lx
                + np.cos(el) * np.cos(az) * ly
                + np.sin(el) * lz)
        elif frame == "NED":
            z_tel = (np.cos(el) * np.cos(az) * lx
                + np.cos(el) * np.sin(az) * ly
                - np.sin(el) * lz)           # Up = -lz in NED
        else:
            raise ValueError("frame must be 'NWU', 'ENU', or 'NED'")
        
        z_tel = self.normalize(z_tel)

        # BUG 4 FIX: always use local Up (frame-independent) so roll reference
        # is consistent and NED doesn't use Down as up_ref → near-zenith singularity
        up_ref = self.normalize(site_ecef)
        x_tel = self.normalize(np.cross(up_ref, z_tel))   # right-hand: Y=Up when el=0
        y_tel = np.cross(z_tel, x_tel)

        R_ecef_tel = np.vstack((x_tel, y_tel, z_tel))
        q = R.from_matrix(R_ecef_tel).as_quat()            # CAUTION: [x,y,z,w] scipy convention 

        # # Boresight direction in ECEF
        # z = (np.cos(el) * np.cos(az) * lx +
        #     np.cos(el) * np.sin(az) * ly +
        #     np.sin(el) * lz )
        # z = self.normalize(z)

        # Telescope body frame
        # Keep x perpendicular to 'up' direction
        # up_ref = lz
        # x = self.normalize(np.cross(z, up_ref))
        # y = np.cross(z, x)

        # Rotation matrix: ECEF → telescope
        # R_ecef_tel = np.vstack((x, y, z))
        # q = R.from_matrix(R_ecef_tel).as_quat()  # [x,y,z,w]

        return np.array([q[3], q[0], q[1], q[2]])  # make scipy convention to scalar-first
    
    def run_phase_calculation(self):
        """Run the Sun postion calculation and plot in Plotly"""

        # Load Skyfield data
        ts = skyapi.load.timescale()
        eph = skyapi.load('de421.bsp')
        sun, moon, earth = eph['sun'], eph['moon'], eph['earth']

        # *Munich* location (hardcoded for now, can be made configurable later)
        munich = eph['earth'] + skyapi.wgs84.latlon(+48.13743, +11.57549)

        # Get time range from UI inputs
        start_date_q = self.start_date_edit.date()
        end_date_q = self.end_date_edit.date()

        start_date_dt = start_date_q.toPyDate()
        end_date_dt = end_date_q.toPyDate()

        t0 = ts.utc(start_date_dt.year, start_date_dt.month, start_date_dt.day)
        tf = ts.utc(end_date_dt.year, end_date_dt.month, end_date_dt.day)

        # Initialize lists to store data for plotting
        times_for_plot = []
        illumination_for_plot = []
        alt_for_plot = []
        az_for_plot = []

        # Get celestial body
        selected_body = self.celes_item.currentText()        
        
        # Calculate for every hour
        current_time = t0
        while current_time.utc_datetime() <= tf.utc_datetime():
            e = munich.at(current_time)
            s = e.observe(sun).apparent()
            m = e.observe(moon).apparent()
            
            if selected_body == 'Sun':
                altitude_obj, azimuth_obj, distance_obj = s.altaz()
                n_rows = 1

            elif selected_body == 'Moon':
                altitude_obj, azimuth_obj, distance_obj = m.altaz()
                percent = 100.0 * m.fraction_illuminated(sun)
                illumination_for_plot.append(percent)
                n_rows = 3

            # Get altaz values
            altitude = altitude_obj.degrees
            azimuth = azimuth_obj.degrees

            # Store data for plotting
            times_for_plot.append(current_time.astimezone(CET))            
            alt_for_plot.append(altitude)
            az_for_plot.append(azimuth)
            current_time += timedelta(hours=1)

        # Create the Plotly figure with subplots
        # fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
        #                     subplot_titles=(f'{selected_body} Altitude/Azimuth'))
        fig = make_subplots(
                        rows=3, cols=1,
                        specs=[
                            [{"type": "xy"}],   # row 1: Alt/Az
                            [{"type": "xy"}],   # row 2: Illumination
                            [{"type": "geo"}]   # row 3: Map
                        ],
                        shared_xaxes=True,      # rows 1 and 2 will share x
                        vertical_spacing=0.08   # optional: space between plots
                    )

        # Disable x-axis for row 3 (geo)
        fig.update_xaxes(showticklabels=False, row=3, col=1)

        # Altitude
        fig.add_trace(go.Scatter(x=times_for_plot, y=alt_for_plot, mode='lines+markers', name='Altitude', 
                                    marker=dict(size=4), line=dict(color='green')), row=1, col=1)
        
        # Add horizontal line at Altitude 0°
        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                        annotation_text="Altitude 0°", annotation_position="bottom right", row=1, col=1)

        # Azimuth
        fig.add_trace(go.Scatter(x=times_for_plot, y=az_for_plot, mode='lines+markers', name='Azimuth', 
                                    marker=dict(size=4), line=dict(color='purple')), row=1, col=1)
        fig.update_yaxes(title_text="Angle(°)", row=1, col=1)

        # Update x-axis to show date and 24hr format, and ensure tick labels are visible
        fig.update_xaxes(tickformat="%Y-%m-%d %H:%M", showticklabels=True, row=1, col=1)
        fig.update_xaxes(tickformat="%Y-%m-%d %H:%M", showticklabels=True, row=1, col=1)
        
        if n_rows>1:
            # Plot : Moon Illumination
            fig.add_trace(go.Scatter(x=times_for_plot, y=illumination_for_plot, mode='lines+markers', name='Illumination', 
                                    marker=dict(size=4)), row=2, col=1)
            fig.update_yaxes(title_text="Moon Illumination (%)", row=2, col=1)
            fig.update_xaxes(showticklabels=True, row=2, col=1) # Ensure x-axis labels are shown for the top plot
            # Highlight illumination above 75%
            high_illumination_times = []
            high_illumination_values = []
            for i, percent in enumerate(illumination_for_plot):
                if percent >= 75:
                    high_illumination_times.append(times_for_plot[i])
                    high_illumination_values.append(percent)

            if high_illumination_times:
                fig.add_trace(go.Scatter(x=high_illumination_times, y=high_illumination_values, mode='markers', name='Illumination >= 75%',
                                        marker=dict(color='gold', size=8, symbol='circle', line=dict(width=1, color='DarkSlateGrey'))), row=2, col=1)
            
            ##------------------------map projection------------------------
            # --- Grid ---
            lats = np.linspace(-90, 90, 181)
            lons = np.linspace(-180, 180, 361)
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            illum_map = np.full_like(lon_grid, np.nan, dtype=float)

            map_time = current_time
            e_map = munich.at(map_time)
            m_map = e_map.observe(moon).apparent()
            illum_global = 100.0 * m_map.fraction_illuminated(sun)

            moon_alt_map = np.full(lon_grid.shape, np.nan, dtype=float)

            # --- Loop over rows only (vectorized over longitudes) ---
            for i in tqdm(range(len(lats)), desc="Computing rows"):
                # Create list of observers for this latitude
                #observers = [earth + skyapi.wgs84.latlon(lats[i], lon) for lon in lons]
                # Compute Moon/Sun altitudes for the row
                #alt_moons = []
                #alt_suns = []
                for j,lon in enumerate(lons):
                    obs = earth + skyapi.wgs84.latlon(float(lats[i]), float(lon)) 
                    t_obs = obs.at(map_time)
                    #m_topo = obs.at(map_time).observe(moon).apparent()
                    #s_topo = obs.at(map_time).observe(sun).apparent()
                    #alt_moon, _, _ = m_topo.altaz()
                    #alt_sun, _, _ = s_topo.altaz()
                    alt_moon = t_obs.observe(moon).apparent().altaz()[0].degrees
                    alt_sun  = t_obs.observe(sun ).apparent().altaz()[0].degrees

                    if alt_moon > 0 and alt_sun < -0:          # moon above horizon & civil darkness
                        # gradient: brighter = higher moon + better illuminated
                        moon_alt_map[i, j] = alt_moon * (illum_global / 100.0)
                    else:
                        moon_alt_map[i, j] = np.nan# 0.0#NaN

                #     alt_moons.append(alt_moon.degrees)
                #     alt_suns.append(alt_sun.degrees)
                
                # alt_moons = np.array(alt_moons)
                # alt_suns = np.array(alt_suns)
                
                # Apply visibility constraints
                # visible = (alt_moons > 5) & (alt_suns < -3)
                # illum_map[i, visible] = illum_global
        
                    # Moon visible and night-time
                    # if alt_moon.degrees > 5 and alt_sun.degrees < -3:
                    #     illum_map[i, j] = illum_global
                    # else:
                    #     illum_map[i, j] = 0.0
                        
            # --- Plot with Heatmap + Geo projection ---
            
            # --- Add as Scattergeo to your existing subplot row 3 ---
            
            # fig.add_trace(go.Heatmap(
            #     z=illum_map,
            #     x=lons,
            #     y=lats,
            #     colorscale='Viridis',
            #     zmin=0,
            #     zmax=100,
            #     colorbar=dict(title="Moon Illumination (%)")
            # ), row=3, col=1)


            # fig.update_geos(
            #     projection_type="natural earth",
            #     showcountries=True,
            #     showcoastlines=True,
            #     showland=True,
            #     landcolor="lightgray",
            #     showocean=True,
            #     oceancolor="rgb(200, 230, 255)",
            #     row=3, col=1
            # )
            # ── Flatten & mask NaNs for Scattergeo ──────────────────────────────────────
            flat_lats = lat_grid.flatten()
            flat_lons = lon_grid.flatten()
            flat_val  = moon_alt_map.flatten()
            mask      = ~np.isnan(flat_val) & (flat_val > 0)

            # ── Add to geo subplot (Scattergeo, NOT Heatmap) ────────────────────────────
            fig.add_trace(
                go.Scattergeo(
                    lat=flat_lats[mask],
                    lon=flat_lons[mask],
                    mode="markers",
                    marker=dict(
                        color=flat_val[mask],
                        colorscale="YlOrRd",          # yellow (low) → orange → red (high)
                        cmin=0,
                        cmax=90 * (illum_global / 100),  # max possible value
                        size=4,
                        opacity=0.75,
                        colorbar=dict(
                            title=f"Moon visibility<br>(alt × illum,  illum={illum_global:.1f}%)",
                            thickness=15,
                            x=1.01,
                        ),
                    ),
                    name="Moon visible",
                    showlegend=False,
                    hovertemplate=(
                        "Lat: %{lat:.1f}°, Lon: %{lon:.1f}°<br>"
                        f"Moon illum: {illum_global:.1f}%<br>"
                        "Score: %{marker.color:.1f}<extra></extra>"
                    ),
                ),
                row=3, col=1,
            )
            # ── Style the geo axes ───────────────────────────────────────────────────────
            fig.update_geos(
                projection_type="natural earth",
                showcountries=True,   showcoastlines=True,
                showland=True,        landcolor="rgb(45, 50, 60)",
                showocean=True,       oceancolor="rgb(20, 30, 55)",
                showlakes=True,       lakecolor="rgb(20, 30, 55)",
                bgcolor="rgb(10, 15, 30)",
                row=3, col=1,
            )


        # Update overall layout
        fig.update_layout(
            height=900, 
            title_text=f"Hourly {selected_body} Data for Munich: ({start_date_dt.strftime('%Y-%m-%d')} to {end_date_dt.strftime('%Y-%m-%d')})",
            title_x=0.5,
            hovermode="x unified"
        )

        # Convert to HTML and display
        html = fig.to_html(include_plotlyjs='cdn')
        self.moon_plot_widget.setHtml(html)
        
        output_dir = os.path.join(self.outputdir, 'figures')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, html)#f'{selected_body}_Data.html')
        print(f"Full-resolution figure saved successfully to {output_file}")

    def propagate_tle(self, tle_line1: str, tle_line2: str, times: Time, output_frame: str = "ECI"):
        """
        Propagate a satellite TLE to ECI (GCRS) or ECEF (ITRS) for given times.

        Parameters
        ----------
        tle_line1 : str
            First line of the TLE.
        tle_line2 : str
            Second line of the TLE.
        times : astropy.time.Time
            Array of observation times (UTC or GPS; Astropy handles scale).
        output_frame : str, optional
            Output reference frame:
                - "ECI"  → GCRS (default)
                - "ECEF" → ITRS

        Returns
        -------
        times_gps_s : np.ndarray, shape (N,)
            GPS time in seconds.
        r_out : np.ndarray, shape (N,3)
            Position vectors [m].
        v_out : np.ndarray, shape (N,3)
            Velocity vectors [m/s].

        Notes
        -----
        - SGP4 outputs TEME.
        - TEME → GCRS conversion is handled rigorously by Astropy.
        - ECEF output uses ITRS (Earth-fixed, rotating frame).
        """
                # Convert times to astropy Time objects
        
        # times = Time(times)
        if not isinstance(times, Time):
            raise TypeError("times must be an astropy.time.Time object")

            # ---- normalize scalar → array ----
        scalar_input = False
        if times.isscalar:
            times = Time([times])
            scalar_input = True
            
        output_frame = output_frame.upper()
        if output_frame not in ("ECI", "ECEF"):
            raise ValueError("output_frame must be 'ECI' or 'ECEF'")

        sat = Satrec.twoline2rv(tle_line1, tle_line2)
        N = len(times)
        r_out = np.zeros((N, 3))
        v_out = np.zeros((N, 3))

        # GPS seconds 
        #times_gps_s = times.gps

        for i, t in enumerate(times):
            err, r_teme_km, v_teme_km_s = sat.sgp4(t.jd1, t.jd2)
            if err != 0:
                raise RuntimeError(f"SGP4 propagation error at {t.iso}")

            # TEME state
            pos = CartesianRepresentation(r_teme_km * u.km)
            vel = CartesianDifferential(v_teme_km_s * u.km / u.s)
            # pos  = CartesianRepresentation(r_teme_km[0]*u.km, r_teme_km[1]*u.km, r_teme_km[2]*u.km)
            # vel  = CartesianDifferential(v_teme_km_s[0]*u.km/u.s, v_teme_km_s[1]*u.km/u.s, v_teme_km_s[2]*u.km/u.s)
            teme = TEME(pos.with_differentials(vel), obstime=t)

            # TEME → GCRS (ECI) (IAU rotation matrices, mean equinox-> celestial ref, time)
            # gcrs = teme.transform_to(GCRS(obstime=t))

            if output_frame == "ECI":
                gcrs = teme.transform_to(GCRS(obstime=t))
                r = gcrs.cartesian.xyz.to_value(u.km)
                v = gcrs.velocity.d_xyz.to_value(u.km / u.s)
                # v = gcrs.cartesian.differentials['s'].d_xyz.to(u.km/u.s).value

            else:  # ECEF (Earth Rotation Angle (ERA), UT1-UTC, Polar motion)
                itrs = gcrs.transform_to(ITRS(obstime=t))
                r = itrs.cartesian.xyz.to_value(u.km)
                v = itrs.velocity.d_xyz.to_value(u.km / u.s)
                # v = itrs.cartesian.differentials['s'].d_xyz.to(u.km/u.s).value


            r_out[i] = r
            v_out[i] = v
        # ---- unwrap scalar outputs ----
        if scalar_input:
            return times.gps[0], r_out[0], v_out[0]
        
        return times.gps, r_out, v_out
    
    def plot_pe_from_csv(self):
        """Plot PE from CSV for the selected settings, roll, pitch, yaw, update_rate, latency."""

        settings_name = self.settings_combo.currentText()
        roll = float(self.roll.value())
        pitch = float(self.pitch.value())
        yaw = float(self.yaw.value())
        update_rate = int(self.update_rate.currentData())
        latency = int(self.latency.currentData())
        active_interpolator = self.active_interpolator

        output_dir = os.path.join(self.outputdir, 'tables', f'{settings_name}_quatpred')
        output_file = os.path.join(
            output_dir,
            f'true_quat{settings_name}_roll{roll}_pitch{pitch}_yaw{yaw}_{update_rate}Hz_{latency}s_{active_interpolator}.csv'
        )
        if not os.path.exists(output_file):
            print(f'CSV file does not exist: {output_file}')
            return
        self.pe_data = self._load_pe_csv(output_file)
        self.update_pe_visualization()
        pe_tab_index = self.graphics_tabs.indexOf(self.pe_graphics)
        if pe_tab_index != -1:
            self.graphics_tabs.setCurrentIndex(pe_tab_index)
    
    def eci_to_ecef(self, times, xs, ys, zs):
        """
        Convert arrays of ECI (J2000) positions to ECEF using astropy for each time step.
        times: array-like, seconds since epoch (UNIX time)
        xs, ys, zs: arrays of ECI positions in meters
        Returns: arrays of ECEF xs, ys, zs
        """
        ecef_xs, ecef_ys, ecef_zs = [], [], []
        for t, x, y, z in zip(times, xs, ys, zs):
            obstime = Time(t, format='unix')
            gcrs = GCRS(CartesianRepresentation(x*u.m, y*u.m, z*u.m), obstime=obstime)
            itrs = gcrs.transform_to(ITRS(obstime=obstime))
            ecef_x, ecef_y, ecef_z = itrs.cartesian.xyz.value
            ecef_xs.append(ecef_x)
            ecef_ys.append(ecef_y)
            ecef_zs.append(ecef_z)
        return np.array(ecef_xs), np.array(ecef_ys), np.array(ecef_zs)

    def site_ecef(self, lat, lon, alt):
        # WGS-84
        a = 6378137.0
        e2 = 6.69437999014e-3

        sl = np.sin(lat)
        cl = np.cos(lat)

        N = a / np.sqrt(1 - e2 * sl**2)

        x = (N + alt) * cl * np.cos(lon)
        y = (N + alt) * cl * np.sin(lon)
        z = (N * (1 - e2) + alt) * sl

        return np.array([x, y, z])

    def _load_pe_csv(self, csv_path):

        df = pd.read_csv(csv_path)
        # Reconstruct arrays for plotting
        time = df['time'].values
        # t_key = df['t_key'].values
        # t_eval = df['t_eval'].values
        # q_key  =  df['q_key'].values
        pe = df['pe'].values
        quat_error = df['quat_error'].values
        q_true = np.vstack([df['q_true_w'].values, df['q_true_x'].values, df['q_true_y'].values, df['q_true_z'].values]).T
        q_pred = np.vstack([df['q_pred_w'].values, df['q_pred_x'].values, df['q_pred_y'].values, df['q_pred_z'].values]).T
        t_from_0 = time - time[0]
        # For t_stamps_updates and data_full_att_h, just use empty arrays (or could reconstruct if saved)
        return {
            'time': time,
            # 'time_key': t_key,
            # 't_eval' :t_eval,
            # 'q_key': q_key,
            'pe': pe,
            'quat_error':quat_error,
            'q_true': q_true,
            'q_pred': q_pred,
            't_from_0': t_from_0,
            't_stamps_updates': np.array([]),
            'data_full_att_h': np.array([])
        }



def main(run_gui):
    # QtWebEngine.initialize()

    if run_gui:
        app = QApplication(sys.argv)

        # --- Splash ---
        splash_pix = QPixmap(400, 200)
        splash_pix.fill(Qt.darkBlue)

        painter = QPainter(splash_pix)
        painter.setPen(QColor(Qt.white))
        painter.setFont(QFont('Arial', 20))
        painter.drawText(splash_pix.rect(), Qt.AlignCenter, "ASTRAA Loading...")
        painter.end()

        splash = QSplashScreen(splash_pix)
        splash.show()
        splash.showMessage(
            "ASTRAA Loading... Initializing GUI",
            Qt.AlignBottom | Qt.AlignCenter,
            Qt.white
        )
        app.processEvents()

        
        main_window = None

        def show_main_window():
            nonlocal main_window     
            main_window = AstraaGUI()
            main_window.show()
            splash.finish(main_window)

        QTimer.singleShot(3000, show_main_window)

        sys.exit(app.exec_())



def run_app() -> None:
    """Console entry point for the installed ``astraa`` command."""
    main(run_gui=True)


if __name__ == "__main__":
    run_app()


