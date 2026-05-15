"""
SGVT-FD Configuration
"""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CRWU_ROOT = os.path.join(DATA_ROOT, "CRWU")
MFPT_ROOT = os.path.join(DATA_ROOT, "MFPT")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "Qwen2.5-VL-3B-Instruct")
MODEL_7B_ROOT = os.path.join(PROJECT_ROOT, "Qwen2.5-VL-7B-Instruct")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results")
FIGS_ROOT = os.path.join(PROJECT_ROOT, "figs")

# Signal processing
SAMPLING_RATE_CRWU = 12000  # Hz
SAMPLING_RATE_MFPT = 97656  # Hz (varies by file, but common)
SIGNAL_LENGTH = 8192  # Fixed length for each sample (power of 2 for FFT)
STFT_WINDOW = 256
STFT_OVERLAP = 128
STFT_NFFT = 512

# Spectrogram
SPEC_HEIGHT = 224
SPEC_WIDTH = 224

# CRWU fault characteristic frequencies (at 1750 RPM, 12kHz sampling)
# These are approximate - actual frequencies depend on RPM
CRWU_FAULT_FREQS = {
    "BPFO": 107.3,  # Ball Pass Frequency Outer
    "BPFI": 162.2,  # Ball Pass Frequency Inner
    "FTF": 14.1,    # Fundamental Train Frequency
    "BSF": 70.6,    # Ball Spin Frequency
}

# CRWU dataset
CRWU_CLASSES = ["Normal", "Ball", "InnerRace", "OuterRace"]
CRWU_CLASS_MAP = {name: i for i, name in enumerate(CRWU_CLASSES)}
CRWU_FAULT_SIZES = ["0007", "0014", "0021", "0028"]
CRWU_LOADS = ["0", "1", "2", "3"]

# MFPT dataset
MFPT_CLASSES = ["Baseline", "OuterRace", "InnerRace"]
MFPT_CLASS_MAP = {name: i for i, name in enumerate(MFPT_CLASSES)}

# SGVT parameters
SGVT_NUM_GROUPS = 32  # Number of semantic groups (K)
SGVT_PATCH_SIZE = 16  # ViT patch size
SGVT_FEATURE_DIM = 768  # CLIP ViT feature dimension
SGVT_MERGED_DIM = 768  # Dimension after merging

# VLM parameters
VLM_MAX_NEW_TOKENS = 512
VLM_TEMPERATURE = 0.1
VLM_PROMPT_TEMPLATE = "Classify this vibration signal spectrogram into one of the following fault types: {classes}. The fault type is:"

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
SEED = 42

# LoRA
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Evaluation
NUM_SEEDS = 3
TEST_RATIO = 0.2
VAL_RATIO = 0.1

# Visualization
FIG_DPI = 300
FIG_FONT_SIZE = 16
FIG_AXES_SIZE = 18
FIG_TITLE_SIZE = 20

# Device
DEVICE = "cuda"
CUDA_VISIBLE_DEVICES = "0"
