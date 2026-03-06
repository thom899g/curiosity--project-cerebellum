#!/usr/bin/env python3
"""
Project Cerebellum - System Vitals Monitoring Daemon
Core persistent daemon for monitoring process trees and predicting resource exhaustion
"""

import asyncio
import json
import logging
import signal
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import psutil
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/cerebellum.log')
    ]
)
logger = logging.getLogger(__name__)

class ProcessPriority(Enum):
    """Process priority classification"""
    CORE_STRATEGIC = 1      # Trading, decision engines
    CORE_OPERATIONAL = 2    # Data ingestion, APIs
    BACKGROUND = 3          # Analytics, reports
    RENDERER = 4           # UI, visualization
    MAINTENANCE = 5        # Cleanup, backups

class ActionType(Enum):
    """Types of actions Cerebellum can take"""
    SUSPEND = "suspend"
    RESUME = "resume"
    TERMINATE = "terminate"
    THROTTLE = "throttle"
    NO_ACTION = "no_action"

@dataclass
class ProcessMetrics:
    """Metrics collected for a process"""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    io_read_bytes: int
    io_write_bytes: int
    num_threads: int
    num_fds: int
    create_time: float
    priority: ProcessPriority
    children: List[int]
    cmdline: List[str]

@dataclass
class SystemState:
    """Current system state snapshot"""
    timestamp: datetime
    total_cpu_percent: float
    total_memory_percent: float
    available_memory_mb: float
    swap_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    processes: Dict[int, ProcessMetrics]
    predicted_exhaustion_minutes: Optional[float] = None
    recommended_actions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.recommended_actions is None:
            self.recommended_actions = []

@dataclass
class ResourceThresholds:
    """Configurable resource thresholds"""
    memory_critical: float = 90.0  # Memory % when critical
    memory_warning: float = 75.0   # Memory % when warning
    cpu_critical: float = 95.0     # CPU % when critical
    cpu_warning: float = 80.0      # CPU % when warning
    swap_critical: float = 80.0    # Swap % when critical
    max_process_memory_mb: float = 4096  # Max MB per process
    renderer_memory_limit_mb: float = 512  # Limit for renderers

class ResourcePredictor:
    """Predicts resource exhaustion using ML models"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = model_path or "cerebellum_predictor.pkl"
        
        # Load existing model if available
        if Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model, self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info(f"Loaded trained model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
    
    def extract_features(self, system_state: SystemState) -> np.ndarray:
        """Extract features from system state for prediction"""
        features = []
        
        # Memory features
        features.append(system_state.total_memory_percent)
        features.append(system_state.available_memory_mb)
        features.append(system_state.swap_percent)
        
        # CPU features
        features.append(system_state.total_cpu_percent)
        
        # Process count features
        features.append(len(system_state.processes))
        
        # Memory usage by priority
        for priority in ProcessPriority:
            priority_memory = sum(
                p.memory_percent for p in system_state.processes.values()
                if p.priority == priority
            )
            features.append(priority_memory)
        
        # Rate of change features (placeholder - would need historical data)
        features.extend([0, 0, 0])  # Placeholders for rates
        
        return np.array(features).reshape(1, -1)
    
    def predict_exhaustion(self, system_state: SystemState) -> Optional[float]:
        """Predict