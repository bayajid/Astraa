<!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cesium.com/downloads/cesiumjs/releases/1.106/Build/Cesium/Cesium.js"></script>
                    <link href="https://cesium.com/downloads/cesiumjs/releases/1.106/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
                    <style>
                        html, body, #cesiumContainer {{
                            width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden;
                        }}
                        #controls {{
                            position: absolute;
                            bottom: 10px;
                            right: 10px;
                            background: rgba(42, 42, 42, 0.8);
                            padding: 10px;
                            border-radius: 5px;
                            color: white;
                            font-family: sans-serif;
                            font-size: 12px;
                            z-index: 1000;
                            max-width: 300px;
                        }}
                        #controls label {{
                            display: block;
                            margin: 5px 0;
                        }}
                        #elevationData {{
                            position: absolute;
                            top: 10px;
                            left: 10px;
                            background: rgba(42, 42, 42, 0.8);
                            padding: 10px;
                            border-radius: 5px;
                            color: white;
                            font-family: sans-serif;
                            font-size: 11px;
                            z-index: 1000;
                            max-height: 300px;
                            overflow-y: auto;
                            max-width: 250px;
                        }}
                        .elevation-item {{
                            margin: 2px 0;
                            padding: 2px;
                        }}
                    </style>
                </head>
                <body>
                    <div id="cesiumContainer"></div>
                    <div id="controls">
                        <label><input type="checkbox" id="showLinks" checked> Show Inter-Satellite Links</label>
                        <label><input type="checkbox" id="showLabels" checked> Show Labels</label>
                        <label><input type="checkbox" id="showElevation" checked> Show Elevation Data</label>
                        <label>Min Elevation: <input type="range" id="minElevation" min="0" max="90" value="5" style="width: 80px;"> <span id="elevationValue">5°</span></label>
                    </div>
                    <div id="elevationData"></div>
                    <script>
                        // Your actual data injected from Python
                        var n_sats = {n_sats};
                        var n_times = {n_times};
                        var sat_names = {json.dumps(sat_names)};
                        var ecef_states = JSON.parse('{json.dumps(ecef_array_list)}');
                        var times_iso = {json.dumps(times_iso)};
                        

                        // Initialize Cesium viewer
                        var viewer = new Cesium.Viewer('cesiumContainer', {{
                            timeline: true,
                            animation: true,
                            terrainProvider: Cesium.createWorldTerrain()
                        }});
                        
                        var scene = viewer.scene;
                        viewer.scene.globe.enableLighting = true;
                        
                        var colors = [
                            Cesium.Color.RED,
                            Cesium.Color.GREEN,
                            Cesium.Color.BLUE,
                            Cesium.Color.YELLOW,
                            Cesium.Color.ORANGE,
                            Cesium.Color.PINK,
                            Cesium.Color.CYAN,
                            Cesium.Color.MAGENTA,
                            Cesium.Color.LIME,
                            Cesium.Color.VIOLET
                        ];

                        var start = Cesium.JulianDate.fromIso8601(times_iso[0]);
                        var stop = Cesium.JulianDate.fromIso8601(times_iso[times_iso.length - 1]);
                        
                        viewer.clock.startTime = start.clone();
                        viewer.clock.stopTime = stop.clone();
                        viewer.clock.currentTime = start.clone();
                        viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
                        viewer.timeline.zoomTo(start, stop);

                        // Create satellite entities
                        var satellites = [];
                        
                        for (var ii = 0; ii < n_sats; ii++) {{
                            var property = new Cesium.SampledPositionProperty();
                            
                            for (var j = 0; j < n_times; j++) {{
                                var idx = ii * 3;
                                var time = Cesium.JulianDate.fromIso8601(times_iso[j]);
                                var position = Cesium.Cartesian3.fromElements(
                                    ecef_states[j][idx],
                                    ecef_states[j][idx+1],
                                    ecef_states[j][idx+2]
                                );
                                property.addSample(time, position);
                            }}
                            
                            var entity = viewer.entities.add({{
                                id: 'sat-' + sat_names[ii],
                                name: sat_names[ii],
                                availability: new Cesium.TimeIntervalCollection([new Cesium.TimeInterval({{
                                    start: start,
                                    stop: stop
                                }})]),
                                position: property,
                                // Use GLB model
                                model: {{
                                    uri: 'http://localhost:8000/examples/resources/sat/ICESat-2.glb',
                                    minimumPixelSize: 128,
                                    maximumScale: 10000,
                                    scale: 1.0,
                                    color: colors[ii % colors.length]
                                }},
                                // Keep point as fallback in case model doesn't load
                                point: {{
                                    pixelSize: 8,
                                    color: colors[ii % colors.length],
                                    outlineColor: Cesium.Color.WHITE,
                                    outlineWidth: 1,
                                    heightReference: Cesium.HeightReference.NONE,
                                    show: false // Hide point when model is visible
                                }},
                                label: {{
                                    text: sat_names[ii],
                                    font: 'bold 14px sans-serif',
                                    fillColor: Cesium.Color.YELLOW,
                                    outlineColor: Cesium.Color.BLACK,
                                    outlineWidth: 2,
                                    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                                    verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                                    pixelOffset: new Cesium.Cartesian2(0, -40),
                                    show: true
                                }},
                                path: {{
                                    resolution: 1,
                                    material: colors[ii % colors.length].withAlpha(0.7),
                                    width: 3,
                                    leadTime: 0,
                                    trailTime: 3600 // 1 hour trail
                                }}
                            }});
                            
                            satellites.push(entity);
                        }}
                        // Inter-satellite link entities
                        var links = [];
                    
                        // Create all possible links between satellites
                        for (var i = 0; i < n_sats; i++) {{
                            for (var j = i + 1; j < n_sats; j++) {{
                                var link = viewer.entities.add({{
                                    id: 'link-' + i + '-' + j,
                                    polyline: {{
                                        positions: new Cesium.CallbackProperty(function(i, j) {{
                                            return function(time, result) {{
                                                var pos1 = satellites[i].position.getValue(time);
                                                var pos2 = satellites[j].position.getValue(time);
                                            
                                                if (pos1 && pos2) {{
                                                    // Check visibility conditions
                                                    if (areSatellitesVisible(pos1, pos2, time)) {{
                                                        return [pos1, pos2];
                                                    }}
                                                }}
                                                return [];
                                            }};
                                        }}(i, j), false),
                                        width: 2,
                                        material: new Cesium.PolylineDashMaterialProperty({{
                                            color: Cesium.Color.CYAN.withAlpha(0.8),
                                            dashLength: 20,
                                            dashPattern: parseInt('1111000011110000', 2)
                                        }}),
                                        clampToGround: false,
                                        followSurface: false,
                                        arcType: Cesium.ArcType.NONE,
                                        show: true
                                    }}
                                }});
                                links.push(link);
                            }}
                        }}



                        //--------------------------------------------------------------
                        // Visibility calculation functions
                        function areSatellitesVisible(pos1, pos2, time) {{
                            if (!pos1 || !pos2) return false;
                            
                            // Check line of sight (no Earth obstruction)
                            if (!hasLineOfSight(pos1, pos2)) return false;
                            
                            // Check minimum elevation angle
                            var minElevation = parseFloat(document.getElementById('minElevation').value) || 5;
                            if (!hasMinimumElevation(pos1, pos2, minElevation)) return false;
                            
                            return true;
                        }}
                        
                        function hasLineOfSight(pos1, pos2) {{
                            // Simple Earth obstruction check
                            var earthRadiusSquared = 6371000 * 6371000; // Earth radius squared
                            
                            // Vector from pos1 to pos2
                            var direction = Cesium.Cartesian3.subtract(pos2, pos1, new Cesium.Cartesian3());
                            var distance = Cesium.Cartesian3.magnitude(direction);
                            
                            if (distance === 0) return false;
                            
                            // Normalize direction vector
                            Cesium.Cartesian3.divideByScalar(direction, distance, direction);
                            
                            // Check multiple points along the line
                            var steps = Math.max(10, Math.floor(distance / 100000)); // Every 100km
                            
                            for (var i = 1; i < steps; i++) {{
                                var t = i / steps;
                                var point = Cesium.Cartesian3.multiplyByScalar(direction, distance * t, new Cesium.Cartesian3());
                                Cesium.Cartesian3.add(pos1, point, point);
                                
                                var distanceFromCenter = Cesium.Cartesian3.magnitudeSquared(point);
                                if (distanceFromCenter < earthRadiusSquared) {{
                                    return false; // Line passes through Earth
                                }}
                            }}
                            
                            return true;
                        }}
                        
                        function hasMinimumElevation(pos1, pos2, minElevationDeg) {{
                            // Calculate elevation angle from pos1 to pos2
                            var earthCenter = new Cesium.Cartesian3(0, 0, 0);
                            var toSat1 = Cesium.Cartesian3.subtract(pos1, earthCenter, new Cesium.Cartesian3());
                            var linkVector = Cesium.Cartesian3.subtract(pos2, pos1, new Cesium.Cartesian3());
                            
                            // Local horizontal plane at pos1
                            var normal1 = Cesium.Cartesian3.normalize(toSat1, new Cesium.Cartesian3());
                            
                            // Project link vector onto the plane perpendicular to normal1
                            var dot = Cesium.Cartesian3.dot(linkVector, normal1);
                            var projectedLink = Cesium.Cartesian3.multiplyByScalar(normal1, dot, new Cesium.Cartesian3());
                            var horizontalComponent = Cesium.Cartesian3.subtract(linkVector, projectedLink, new Cesium.Cartesian3());
                            
                            if (Cesium.Cartesian3.magnitude(horizontalComponent) === 0) return true;
                            
                            // Calculate elevation angle
                            var elevationRad = Math.atan2(dot, Cesium.Cartesian3.magnitude(horizontalComponent));
                            var elevationDeg = Cesium.Math.toDegrees(Math.abs(elevationRad));
                            
                            return elevationDeg >= minElevationDeg;
                        }}

                        // Get elevation data div
                        var elevationDataDiv = document.getElementById("elevationData");

                        function updateElevationAngles(time) {{
                            // Check if elevation display is enabled
                            if (!document.getElementById('showElevation').checked) {{
                                elevationDataDiv.style.display = 'none';
                                return;
                            }}
                            elevationDataDiv.style.display = 'block';

                            // Clear previous data
                            elevationDataDiv.innerHTML = '<div style="font-weight: bold; margin-bottom: 5px;">Elevation Angles:</div>';

                            // Loop through each satellite pair to calculate and display elevation angles
                            for (var i = 0; i < n_sats; i++) {{
                                for (var j = i + 1; j < n_sats; j++) {{
                                    var pos1 = satellites[i].position.getValue(time);
                                    var pos2 = satellites[j].position.getValue(time);

                                    if (pos1 && pos2) {{
                                        // Calculate the elevation angle between satellites i and j
                                        var earthCenter = new Cesium.Cartesian3(0, 0, 0);
                                        var toSat1 = Cesium.Cartesian3.subtract(pos1, earthCenter, new Cesium.Cartesian3());
                                        var linkVector = Cesium.Cartesian3.subtract(pos2, pos1, new Cesium.Cartesian3());

                                        // Local horizontal plane at pos1
                                        var normal1 = Cesium.Cartesian3.normalize(toSat1, new Cesium.Cartesian3());

                                        // Project link vector onto the plane perpendicular to normal1
                                        var dot = Cesium.Cartesian3.dot(linkVector, normal1);
                                        var projectedLink = Cesium.Cartesian3.multiplyByScalar(normal1, dot, new Cesium.Cartesian3());
                                        var horizontalComponent = Cesium.Cartesian3.subtract(linkVector, projectedLink, new Cesium.Cartesian3());

                                        var elevationDeg = 0;
                                        if (Cesium.Cartesian3.magnitude(horizontalComponent) !== 0) {{
                                            // Calculate the elevation angle in radians
                                            var elevationRad = Math.atan2(dot, Cesium.Cartesian3.magnitude(horizontalComponent));
                                            elevationDeg = Cesium.Math.toDegrees(Math.abs(elevationRad));
                                        }}

                                        // Create HTML elements for each elevation value
                                        var elevationItem = document.createElement("div");
                                        elevationItem.classList.add("elevation-item");
                                        
                                        // Use satellite names instead of indices for clarity
                                        var satName1 = sat_names[i] || ('Sat-' + i);
                                        var satName2 = sat_names[j] || ('Sat-' + j);
                                        
                                        // Check if satellites are visible to each other
                                        var isVisible = areSatellitesVisible(pos1, pos2, time);
                                        var visibilityText = isVisible ? "✓" : "✗";
                                        // var textColor = isVisible ? "#00ff00" : "#ff6666";
                                        var textColor = elevationDeg > 0 ? "#00ff00" : "##ff6666";  // Green if elevation > 0, Red if elevation < 0
                                        
                                        elevationItem.innerHTML = '<span style="color: ' + textColor + '">' + visibilityText + '</span> ' + satName1 + ' ↔ ' + satName2 + ': ' + elevationDeg.toFixed(1) + '°';
                                        elevationDataDiv.appendChild(elevationItem);
                                    }}
                                }}
                            }}
                        }}


                        // Control event handlers
                        document.getElementById('showLinks').addEventListener('change', function(e) {{
                            links.forEach(function(link) {{
                                link.polyline.show = e.target.checked;
                            }});
                        }});
                        
                        document.getElementById('showLabels').addEventListener('change', function(e) {{
                            satellites.forEach(function(sat) {{
                                sat.label.show = e.target.checked;
                            }});
                        }});
                        
                        document.getElementById('showElevation').addEventListener('change', function(e) {{
                            if (!e.target.checked) {{
                                elevationDataDiv.style.display = 'none';
                            }}
                        }});
                        
                        document.getElementById('minElevation').addEventListener('input', function(e) {{
                            document.getElementById('elevationValue').textContent = e.target.value + '°';
                        }});

                        // Update elevation display (throttled to avoid performance issues)
                        var lastUpdate = 0;
                        viewer.scene.preRender.addEventListener(function (scene, time) {{
                            var now = Date.now();
                            if (now - lastUpdate > 100) {{ // Update every 100ms
                                updateElevationAngles(viewer.clock.currentTime);
                                lastUpdate = now;
                            }}
                        }});

                        // Zoom to satellites first, then adjust camera for better Earth view
                        viewer.zoomTo(viewer.entities).then(function() {{
                            // Set a better camera position to show both Earth and satellites
                            var earthCenter = Cesium.Cartesian3.fromDegrees(0, 0, 0);
                            viewer.camera.lookAt(earthCenter, new Cesium.Cartesian3(0, 0, 20000000));
                            viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
                        }});
                    </script>
                </body>
                </html>

