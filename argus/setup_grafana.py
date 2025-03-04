"""
Setup script for Grafana dashboards.

This script creates a Docker Compose configuration for running Grafana locally,
and also generates Grafana dashboard JSON files for LLM metrics visualization.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_docker_compose_file(base_dir: str = "./monitoring",
                             influxdb_data_dir: str = "./influxdb_data",
                             grafana_data_dir: str = "./grafana_data") -> str:
    """
    Create a Docker Compose file for Grafana and InfluxDB.
    
    Args:
        base_dir: Base directory for monitoring setup
        influxdb_data_dir: Directory for InfluxDB data
        grafana_data_dir: Directory for Grafana data
        
    Returns:
        Path to the created Docker Compose file
    """
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    
    compose_file = os.path.join(base_dir, "docker-compose.yml")
    
    # Docker compose content
    content = f"""version: '3'

services:
  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    volumes:
      - {influxdb_data_dir}:/var/lib/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword
      - DOCKER_INFLUXDB_INIT_ORG=llm_metrics
      - DOCKER_INFLUXDB_INIT_BUCKET=llm_metrics
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=mytoken
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - {grafana_data_dir}:/var/lib/grafana
      - ./dashboards:/etc/grafana/provisioning/dashboards
      - ./datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    restart: unless-stopped
    depends_on:
      - influxdb
"""
    
    # Write to file
    with open(compose_file, 'w') as f:
        f.write(content)
    
    logger.info(f"Created Docker Compose file: {compose_file}")
    return compose_file


def create_influxdb_datasource(base_dir: str = "./monitoring") -> str:
    """
    Create a Grafana datasource configuration for InfluxDB.
    
    Args:
        base_dir: Base directory for monitoring setup
        
    Returns:
        Path to the created datasource file
    """
    # Create directories
    datasources_dir = os.path.join(base_dir, "datasources")
    os.makedirs(datasources_dir, exist_ok=True)
    
    datasource_file = os.path.join(datasources_dir, "influxdb.yml")
    
    # Datasource content
    content = """apiVersion: 1

datasources:
  - name: InfluxDB_LLM_Metrics
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    secureJsonData:
      token: mytoken
    jsonData:
      version: Flux
      organization: llm_metrics
      defaultBucket: llm_metrics
      tlsSkipVerify: true
    isDefault: true
"""
    
    # Write to file
    with open(datasource_file, 'w') as f:
        f.write(content)
    
    logger.info(f"Created InfluxDB datasource configuration: {datasource_file}")
    return datasource_file


def create_dashboard_config(base_dir: str = "./monitoring") -> str:
    """
    Create a Grafana dashboard provisioning configuration.
    
    Args:
        base_dir: Base directory for monitoring setup
        
    Returns:
        Path to the created dashboard configuration file
    """
    # Create directories
    dashboards_dir = os.path.join(base_dir, "dashboards")
    os.makedirs(dashboards_dir, exist_ok=True)
    
    dashboard_config_file = os.path.join(dashboards_dir, "dashboards.yml")
    
    # Dashboard configuration content
    content = """apiVersion: 1

providers:
  - name: 'LLM Metrics'
    orgId: 1
    folder: 'LLM Metrics'
    folderUid: ''
    type: file
    disableDeletion: false
    editable: true
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
"""
    
    # Write to file
    with open(dashboard_config_file, 'w') as f:
        f.write(content)
    
    logger.info(f"Created dashboard configuration: {dashboard_config_file}")
    return dashboard_config_file


def create_model_comparison_dashboard(base_dir: str = "./monitoring") -> str:
    """
    Create a Grafana dashboard for comparing LLM models.
    
    Args:
        base_dir: Base directory for monitoring setup
        
    Returns:
        Path to the created dashboard file
    """
    # Create directories
    dashboards_dir = os.path.join(base_dir, "dashboards")
    os.makedirs(dashboards_dir, exist_ok=True)
    
    dashboard_file = os.path.join(dashboards_dir, "model_comparison_dashboard.json")
    
    # Dashboard content - this is a simplified example, real dashboards would be more complex
    dashboard = {
        "annotations": {
            "list": [
                {
                    "builtIn": 1,
                    "datasource": {
                        "type": "grafana",
                        "uid": "-- Grafana --"
                    },
                    "enable": True,
                    "hide": True,
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "name": "Annotations & Alerts",
                    "type": "dashboard"
                }
            ]
        },
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": [
            {
                "datasource": {
                    "type": "influxdb",
                    "uid": "${DS_INFLUXDB_LLM_METRICS}"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {
                            "mode": "palette-classic"
                        },
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False
                            },
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {
                                "type": "linear"
                            },
                            "showPoints": "auto",
                            "spanNulls": False,
                            "stacking": {
                                "group": "A",
                                "mode": "none"
                            },
                            "thresholdsStyle": {
                                "mode": "off"
                            }
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {
                                    "color": "green",
                                    "value": None
                                },
                                {
                                    "color": "red",
                                    "value": 80
                                }
                            ]
                        },
                        "unit": "none"
                    },
                    "overrides": []
                },
                "gridPos": {
                    "h": 9,
                    "w": 12,
                    "x": 0,
                    "y": 0
                },
                "id": 1,
                "options": {
                    "legend": {
                        "calcs": [],
                        "displayMode": "list",
                        "placement": "bottom",
                        "showLegend": True
                    },
                    "tooltip": {
                        "mode": "single",
                        "sort": "none"
                    }
                },
                "title": "Average Response Length by Model",
                "type": "timeseries",
                "targets": [
                    {
                        "datasource": {
                            "type": "influxdb",
                            "uid": "${DS_INFLUXDB_LLM_METRICS}"
                        },
                        "query": """from(bucket: "llm_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "llm_metric")
  |> filter(fn: (r) => r["metric"] == "ResponseLengthMetric")
  |> filter(fn: (r) => r["_field"] == "value")
  |> group(columns: ["model"])
  |> mean()
  |> yield(name: "mean")""",
                        "refId": "A"
                    }
                ]
            },
            {
                "datasource": {
                    "type": "influxdb",
                    "uid": "${DS_INFLUXDB_LLM_METRICS}"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {
                            "mode": "thresholds"
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {
                                    "color": "red",
                                    "value": None
                                },
                                {
                                    "color": "yellow",
                                    "value": 0.3
                                },
                                {
                                    "color": "green",
                                    "value": 0.7
                                }
                            ]
                        },
                        "unit": "percentunit"
                    },
                    "overrides": []
                },
                "gridPos": {
                    "h": 9,
                    "w": 12,
                    "x": 12,
                    "y": 0
                },
                "id": 2,
                "options": {
                    "orientation": "auto",
                    "reduceOptions": {
                        "calcs": [
                            "lastNotNull"
                        ],
                        "fields": "",
                        "values": False
                    },
                    "showThresholdLabels": False,
                    "showThresholdMarkers": True,
                    "text": {}
                },
                "title": "Hallucination Score by Model",
                "type": "gauge",
                "targets": [
                    {
                        "datasource": {
                            "type": "influxdb",
                            "uid": "${DS_INFLUXDB_LLM_METRICS}"
                        },
                        "query": """from(bucket: "llm_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "llm_metric")
  |> filter(fn: (r) => r["metric"] == "HallucinationMetricAdapter")
  |> filter(fn: (r) => r["_field"] == "value")
  |> group(columns: ["model"])
  |> mean()
  |> yield(name: "mean")""",
                        "refId": "A"
                    }
                ]
            },
            {
                "datasource": {
                    "type": "influxdb",
                    "uid": "${DS_INFLUXDB_LLM_METRICS}"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {
                            "mode": "palette-classic"
                        },
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "fillOpacity": 80,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False
                            },
                            "lineWidth": 1,
                            "scaleDistribution": {
                                "type": "linear"
                            },
                            "thresholdsStyle": {
                                "mode": "off"
                            }
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {
                                    "color": "green",
                                    "value": None
                                },
                                {
                                    "color": "red",
                                    "value": 80
                                }
                            ]
                        }
                    },
                    "overrides": []
                },
                "gridPos": {
                    "h": 9,
                    "w": 12,
                    "x": 0,
                    "y": 9
                },
                "id": 3,
                "options": {
                    "barRadius": 0,
                    "barWidth": 0.97,
                    "groupWidth": 0.7,
                    "legend": {
                        "calcs": [],
                        "displayMode": "list",
                        "placement": "bottom",
                        "showLegend": True
                    },
                    "orientation": "auto",
                    "showValue": "auto",
                    "stacking": "none",
                    "tooltip": {
                        "mode": "single",
                        "sort": "none"
                    },
                    "xTickLabelRotation": 0,
                    "xTickLabelSpacing": 0
                },
                "title": "Metrics by Model",
                "type": "barchart",
                "targets": [
                    {
                        "datasource": {
                            "type": "influxdb",
                            "uid": "${DS_INFLUXDB_LLM_METRICS}"
                        },
                        "query": """from(bucket: "llm_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "llm_metric")
  |> filter(fn: (r) => r["_field"] == "value")
  |> group(columns: ["model", "metric"])
  |> mean()
  |> group(columns: ["metric", "model"])
  |> yield(name: "mean")""",
                        "refId": "A"
                    }
                ]
            },
            {
                "datasource": {
                    "type": "influxdb",
                    "uid": "${DS_INFLUXDB_LLM_METRICS}"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {
                            "mode": "thresholds"
                        },
                        "custom": {
                            "align": "auto",
                            "cellOptions": {
                                "type": "auto"
                            },
                            "inspect": False
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {
                                    "color": "green",
                                    "value": None
                                },
                                {
                                    "color": "red",
                                    "value": 80
                                }
                            ]
                        }
                    },
                    "overrides": []
                },
                "gridPos": {
                    "h": 9,
                    "w": 12,
                    "x": 12,
                    "y": 9
                },
                "id": 4,
                "options": {
                    "footer": {
                        "countRows": False,
                        "fields": "",
                        "reducer": [
                            "sum"
                        ],
                        "show": False
                    },
                    "showHeader": True
                },
                "title": "Recent Batch Metrics",
                "type": "table",
                "targets": [
                    {
                        "datasource": {
                            "type": "influxdb",
                            "uid": "${DS_INFLUXDB_LLM_METRICS}"
                        },
                        "query": """from(bucket: "llm_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "llm_metric")
  |> filter(fn: (r) => r["_field"] == "value")
  |> group(columns: ["batch_id", "model", "metric"])
  |> mean()
  |> group(columns: ["batch_id", "model", "metric", "_value"])
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 20)
  |> yield(name: "mean")""",
                        "refId": "A"
                    }
                ]
            }
        ],
        "refresh": "5s",
        "schemaVersion": 38,
        "style": "dark",
        "tags": ["llm", "metrics"],
        "templating": {
            "list": [
                {
                    "current": {
                        "selected": False,
                        "text": "InfluxDB_LLM_Metrics",
                        "value": "InfluxDB_LLM_Metrics"
                    },
                    "hide": 0,
                    "includeAll": False,
                    "label": "Data Source",
                    "multi": False,
                    "name": "DS_INFLUXDB_LLM_METRICS",
                    "options": [],
                    "query": "influxdb",
                    "refresh": 1,
                    "regex": "",
                    "skipUrlSync": False,
                    "type": "datasource"
                }
            ]
        },
        "time": {
            "from": "now-6h",
            "to": "now"
        },
        "timepicker": {},
        "timezone": "",
        "title": "LLM Model Comparison",
        "uid": "llm-model-comparison",
        "version": 1,
        "weekStart": ""
    }
    
    # Write to file
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    logger.info(f"Created model comparison dashboard: {dashboard_file}")
    return dashboard_file


def create_metrics_dashboard(base_dir: str = "./monitoring") -> str:
    """
    Create a Grafana dashboard for LLM metrics.
    
    Args:
        base_dir: Base directory for monitoring setup
        
    Returns:
        Path to the created dashboard file
    """
    # Create directories
    dashboards_dir = os.path.join(base_dir, "dashboards")
    os.makedirs(dashboards_dir, exist_ok=True)
    
    dashboard_file = os.path.join(dashboards_dir, "llm_metrics_dashboard.json")
    
    # Dashboard content - this is a simplified example
    dashboard = {
        "annotations": {
            "list": [
                {
                    "builtIn": 1,
                    "datasource": {
                        "type": "grafana",
                        "uid": "-- Grafana --"
                    },
                    "enable": True,
                    "hide": True,
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "name": "Annotations & Alerts",
                    "type": "dashboard"
                }
            ]
        },
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": [
            {
                "datasource": {
                    "type": "influxdb",
                    "uid": "${DS_INFLUXDB_LLM_METRICS}"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {
                            "mode": "palette-classic"
                        },
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False
                            },
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {
                                "type": "linear"
                            },
                            "showPoints": "auto",
                            "spanNulls": False,
                            "stacking": {
                                "group": "A",
                                "mode": "none"
                            },
                            "thresholdsStyle": {
                                "mode": "off"
                            }
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {
                                    "color": "green",
                                    "value": None
                                },
                                {
                                    "color": "red",
                                    "value": 80
                                }
                            ]
                        },
                        "unit": "none"
                    },
                    "overrides": []
                },
                "gridPos": {
                    "h": 9,
                    "w": 24,
                    "x": 0,
                    "y": 0
                },
                "id": 1,
                "options": {
                    "legend": {
                        "calcs": [],
                        "displayMode": "list",
                        "placement": "bottom",
                        "showLegend": True
                    },
                    "tooltip": {
                        "mode": "single",
                        "sort": "none"
                    }
                },
                "title": "Metrics Over Time",
                "type": "timeseries",
                "targets": [
                    {
                        "datasource": {
                            "type": "influxdb",
                            "uid": "${DS_INFLUXDB_LLM_METRICS}"
                        },
                        "query": """from(bucket: "llm_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "llm_metric")
  |> filter(fn: (r) => r["metric"] == "${metric:raw}")
  |> filter(fn: (r) => r["_field"] == "value")
  |> group(columns: ["model"])
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "mean")""",
                        "refId": "A"
                    }
                ]
            },
            {
                "datasource": {
                    "type": "influxdb",
                    "uid": "${DS_INFLUXDB_LLM_METRICS}"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {
                            "mode": "thresholds"
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {
                                    "color": "green",
                                    "value": None
                                },
                                {
                                    "color": "red",
                                    "value": 80
                                }
                            ]
                        }
                    },
                    "overrides": []
                },
                "gridPos": {
                    "h": 8,
                    "w": 12,
                    "x": 0,
                    "y": 9
                },
                "id": 2,
                "options": {
                    "displayMode": "gradient",
                    "minVizHeight": 10,
                    "minVizWidth": 0,
                    "orientation": "auto",
                    "reduceOptions": {
                        "calcs": [
                            "mean"
                        ],
                        "fields": "",
                        "values": False
                    },
                    "showUnfilled": True,
                    "text": {}
                },
                "title": "Average Metric Value by Model",
                "type": "bargauge",
                "targets": [
                    {
                        "datasource": {
                            "type": "influxdb",
                            "uid": "${DS_INFLUXDB_LLM_METRICS}"
                        },
                        "query": """from(bucket: "llm_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "llm_metric")
  |> filter(fn: (r) => r["metric"] == "${metric:raw}")
  |> filter(fn: (r) => r["_field"] == "value")
  |> group(columns: ["model"])
  |> mean()
  |> yield(name: "mean")""",
                        "refId": "A"
                    }
                ]
            },
            {
                "datasource": {
                    "type": "influxdb",
                    "uid": "${DS_INFLUXDB_LLM_METRICS}"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {
                            "mode": "palette-classic"
                        },
                        "custom": {
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False
                            }
                        },
                        "mappings": []
                    },
                    "overrides": []
                },
                "gridPos": {
                    "h": 8,
                    "w": 12,
                    "x": 12,
                    "y": 9
                },
                "id": 3,
                "options": {
                    "displayLabels": ["name", "value"],
                    "legend": {
                        "displayMode": "list",
                        "placement": "right",
                        "showLegend": True
                    },
                    "pieType": "pie",
                    "reduceOptions": {
                        "calcs": [
                            "lastNotNull"
                        ],
                        "fields": "",
                        "values": False
                    },
                    "tooltip": {
                        "mode": "single",
                        "sort": "none"
                    }
                },
                "title": "Sample Count by Model",
                "type": "piechart",
                "targets": [
                    {
                        "datasource": {
                            "type": "influxdb",
                            "uid": "${DS_INFLUXDB_LLM_METRICS}"
                        },
                        "query": """from(bucket: "llm_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "llm_metric")
  |> filter(fn: (r) => r["metric"] == "${metric:raw}")
  |> filter(fn: (r) => r["_field"] == "sample_count")
  |> group(columns: ["model"])
  |> sum()
  |> yield(name: "sum")""",
                        "refId": "A"
                    }
                ]
            }
        ],
        "refresh": "5s",
        "schemaVersion": 38,
        "style": "dark",
        "tags": ["llm", "metrics"],
        "templating": {
            "list": [
                {
                    "current": {
                        "selected": False,
                        "text": "InfluxDB_LLM_Metrics",
                        "value": "InfluxDB_LLM_Metrics"
                    },
                    "hide": 0,
                    "includeAll": False,
                    "label": "Data Source",
                    "multi": False,
                    "name": "DS_INFLUXDB_LLM_METRICS",
                    "options": [],
                    "query": "influxdb",
                    "refresh": 1,
                    "regex": "",
                    "skipUrlSync": False,
                    "type": "datasource"
                },
                {
                    "current": {
                        "selected": True,
                        "text": "ResponseLengthMetric",
                        "value": "ResponseLengthMetric"
                    },
                    "datasource": {
                        "type": "influxdb",
                        "uid": "${DS_INFLUXDB_LLM_METRICS}"
                    },
                    "definition": """import "influxdata/influxdb/v1"
v1.tagValues(bucket: "llm_metrics", tag: "metric")""",
                    "hide": 0,
                    "includeAll": False,
                    "label": "Metric",
                    "multi": False,
                    "name": "metric",
                    "options": [],
                    "query": """import "influxdata/influxdb/v1"
v1.tagValues(bucket: "llm_metrics", tag: "metric")""",
                    "refresh": 1,
                    "regex": "",
                    "skipUrlSync": False,
                    "sort": 1,
                    "type": "query"
                }
            ]
        },
        "time": {
            "from": "now-6h",
            "to": "now"
        },
        "timepicker": {},
        "timezone": "",
        "title": "LLM Metrics Dashboard",
        "uid": "llm-metrics",
        "version": 1,
        "weekStart": ""
    }
    
    # Write to file
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    logger.info(f"Created metrics dashboard: {dashboard_file}")
    return dashboard_file


def setup_grafana(base_dir: str = "./monitoring") -> None:
    """
    Set up Grafana and InfluxDB with dashboards for LLM metrics.
    
    Args:
        base_dir: Base directory for monitoring setup
    """
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    
    # Create Docker Compose file
    compose_file = create_docker_compose_file(base_dir)
    
    # Create datasource configuration
    datasource_file = create_influxdb_datasource(base_dir)
    
    # Create dashboard configuration
    dashboard_config = create_dashboard_config(base_dir)
    
    # Create dashboards
    model_dashboard = create_model_comparison_dashboard(base_dir)
    metrics_dashboard = create_metrics_dashboard(base_dir)
    
    # Print setup instructions
    print("\n=== Grafana Setup Instructions ===")
    print("1. Start Grafana and InfluxDB with the following command:")
    print(f"   cd {base_dir} && docker-compose up -d")
    print("\n2. Access Grafana at: http://localhost:3000")
    print("   Username: admin")
    print("   Password: admin")
    print("\n3. Access InfluxDB at: http://localhost:8086")
    print("   Username: admin")
    print("   Password: adminpassword")
    print("   Token: mytoken")
    print("\n4. Dashboards will be automatically provisioned:")
    print(f"   - LLM Model Comparison")
    print(f"   - LLM Metrics Dashboard")
    print("\n5. To stop the services:")
    print(f"   cd {base_dir} && docker-compose down")


if __name__ == "__main__":
    setup_grafana()