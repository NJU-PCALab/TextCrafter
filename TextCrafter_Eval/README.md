# TextCrafter Unified Evaluation Tool

A unified evaluation tool for 5 text-to-image generation metrics with one-click execution.

## ✨ Metrics Overview

- **Word Accuracy** - Text recognition accuracy (based on PaddleOCR)
- **NED** - Normalized Edit Distance (based on Levenshtein distance)
- **CLIPScore** - Image-text similarity score (based on CLIP model)
- **VQAScore** - Visual Question Answering score (based on T2V-Metrics)
- **Aesthetic Score** - Aesthetic evaluation score (based on pre-trained aesthetic model)

## 📋 Requirements

- **Operating System**: Linux
- **Python Version**: 3.10.15
- **GPU**: NVIDIA GPU (supporting CUDA 11.8+)
- **GPU Memory**: Recommended 24GB+

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/NJU-PCALab/TextCrafter.git
cd TextCrafter_Eval
```

### Step 2: Create Conda Environment

```bash
# Create basic environment (Python and system libraries only)
conda env create -f unified_environment.yml
```

### Step 3: Activate Environment

```bash
conda activate textcrafter_eval
```

### Step 4: Install Dependencies

```bash
# Run installation script (includes all Python packages and PaddlePaddle)
bash install_paddle_deps.sh
```

After successful installation, you will see all components marked with ✓. If you see ✗, please check the error messages.


## 📖 Usage

### Basic Usage

```bash
python unified_metrics_eval.py --benchmark_dir /path/to/CVTG-2K --result_dir /path/to/results --output_file results.json
```

### Parameter Description

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--benchmark_dir` | Benchmark data directory | `/path/to/CVTG-2K` |
| `--result_dir` | Generated image results directory | `/path/to/results` |
| `--output_file` | Output results file path | `results.json` |
| `--device` | Computing device (auto/cuda/cpu) | `cuda` |
| `--cache_dir` | HuggingFace auto-download models directory | `/path/to/cache` |
| `--use_hf_mirror` | Use HuggingFace mirror | (enabled by default) |

### Complete Example

```bash
python unified_metrics_eval.py --benchmark_dir /path/to/CVTG-2K --result_dir /path/to/results --output_file results.json --cache_dir /path/to/cache --use_hf_mirror
```

## 📁 Data Format Requirements

### Input Directory Structure (minimum required content)

```
benchmark_dir/
├── CVTG/
│   ├── 2.json
│   ├── 3.json
│   ├── 4.json
│   └── 5.json
└── CVTG-Style/
    ├── 2.json
    ├── 3.json
    ├── 4.json
    └── 5.json

result_dir/
├── CVTG/
│   ├── 2/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   ├── 3/
│   ├── 4/
│   └── 5/
└── CVTG-Style/
    ├── 2/
    ├── 3/
    ├── 4/
    └── 5/
```

Supported image formats: `.png`, `.jpg`, `.jpeg` (case-insensitive).



## ⚠️ Important Notes

### GPU Usage Requirements
- **GPU Required**: All metric calculations require GPU acceleration, CPU mode is not supported
- **Memory Management**: If GPU memory is insufficient, please clear other GPU processes

### Network Requirements
- **Model Download**: First run will automatically download pre-trained models to cache directory
- **Mirror Acceleration**: Uses HuggingFace mirror by default to accelerate downloads
- **Cache Settings**: Recommend setting a sufficiently large cache directory

### Data Requirements
- **Image Formats**: Supports PNG, JPG, JPEG formats
- **File Naming**: Image file names should correspond to the index in JSON
- **Path Structure**: Strictly follow the directory structure described above