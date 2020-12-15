from datetime import datetime
import os

# Create output directory
OUTPUT_ROOT = '../out'
OUTPUT_DIR = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.isdir(OUTPUT_ROOT):
    os.mkdir(OUTPUT_ROOT)
os.mkdir(os.path.join(OUTPUT_ROOT, OUTPUT_DIR))

# Number of non-terminal states
NON_TERMINAL_STATES = 5

# Number of training sets
TRAINING_SETS = 100

# Number of sequences
SEQUENCES = 10

# Actual values for the states
IDEAL_PREDICTIONS = [1.0 / 6, 1.0 / 3, 1.0 / 2, 2.0 / 3, 5.0 / 6]

# Prevents long-running jobs with Max Iterations
MAX_ITERATIONS = 1000

# Controls error by limiting sequence size
MAX_SEQUENCE_SIZE = 8

# Testing
TEST_MODE = False

