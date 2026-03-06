"""
Constants and configuration values for report generation.

This module contains commonly used constants to avoid magic numbers
and improve code maintainability.
"""

# Image processing constants
IMAGE_SIZE = 224
IMAGE_RESIZE_DIM = (IMAGE_SIZE, IMAGE_SIZE)

# Training constants
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_SEQ_LENGTH = 100
DEFAULT_THRESHOLD = 10

# Detection constants
DEFAULT_DETECTION_SCORE_THRESHOLD = 0.7
DEFAULT_DETECTION_SCORE_THRESHOLD_HIGH = 0.9
NUM_DETECTION_CLASSES = 9
MAX_BOXES_PER_IMAGE = 10

# Report generation constants
MAX_REPORT_LENGTH = 77
LABEL_NAME_LENGTH = 5

# Model constants
LOCATION_EMBEDDING_DIM = 256
COORDINATE_SCALE = 100

# Training hyperparameters
GRADIENT_CLIP_VALUE = 0.1
PRINT_INTERVAL = 5000

# EMA (Exponential Moving Average) constants
EMA_DECAY = 0.999
CONSISTENCY_WEIGHT = 0.1
RAMPUP_LENGTH = 200.0

# Monitoring
MONITOR_MODES = ['min', 'max']

