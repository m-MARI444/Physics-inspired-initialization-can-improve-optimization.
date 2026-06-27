# PSSA-GPT Cross-Machine Setup & Resumption Guide

This guide describes how to clone this repository to another computer, install the necessary dependencies (including Git LFS), load the 25k training checkpoint, and resume the campaign or launch the visual dashboard.

---

## 📋 Prerequisites & System Requirements

* **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
* **Python**: Version `3.10` or higher
* **GPU**: CUDA-capable Nvidia GPU (recommended, minimum 4GB VRAM) or CPU
* **Git LFS**: Required to download the 1.5 GB model checkpoint file.

---

## 🚀 Step-by-Step Setup

### Step 1: Install Git LFS
Since the model checkpoint is a large binary file (~1.5 GB), standard Git only clones the lightweight LFS text pointers. You must install Git LFS to download the actual model parameters.

**On Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
```

---

### Step 2: Clone and Fetch the Repository
Clone the repository and explicitly download the large checkpoint files:

```bash
# Clone the repository
git clone https://github.com/m-MARI444/Persistent-Sparse-Semantic-Architecture-PSSA-.git
cd Persistent-Sparse-Semantic-Architecture-PSSA-/pssa_project

# Fetch and download the 1.5 GB model checkpoint
git lfs pull
```

---

### Step 3: Set Up a Python Virtual Environment
Create a clean environment to avoid library conflicts:

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

---

### Step 4: Install Dependencies
Install PyTorch (with CUDA support) and all required packages:

```bash
# 1. Install PyTorch (with CUDA support)
pip3 install torch torchvision torchaudio

# 2. Install all model and campaign dependencies
pip3 install transformers datasets psutil bitsandbytes truststore httpx urllib3
```

> [!NOTE]
> `bitsandbytes` is used to load the 8-bit AdamW optimizer, which saves ~400MB of VRAM during training. If you run on a CPU or a system without CUDA, it will automatically fallback to standard AdamW.

---

## ☁️ Running on Free Cloud GPUs (Kaggle / Colab)

For developing and testing the PSSA architecture, free cloud notebook platforms are highly useful options if your local machine lacks a powerful GPU.

### Option 1: Kaggle Notebooks (Free Tesla T4 - 16GB VRAM or Dual T4 x2)
Kaggle provides approximately 30 free GPU hours per week, which is excellent for prototyping. The training campaign script automatically detects multiple GPUs (if you select GPU T4 x2) and parallelizes training using `nn.DataParallel` out-of-the-box.

1. **Create a Notebook**: Go to [Kaggle](https://www.kaggle.com/), create a new Notebook, and in the Settings panel set the **Accelerator** to **GPU T4** (or T4 x2).
2. **Clone & Pull in a Cell**:
   ```python
   !git clone https://github.com/m-MARI444/Persistent-Sparse-Semantic-Architecture-PSSA-.git
   %cd Persistent-Sparse-Semantic-Architecture-PSSA-/pssa_project
   !git lfs pull
   ```
3. **Install Dependencies**:
   ```python
   !pip install transformers datasets psutil bitsandbytes truststore httpx urllib3
   ```
4. **Run Training**:
   ```python
   !python training/run_scaling_campaign.py --dataset=fineweb
   ```

### Option 2: Google Colab (Free Tesla T4 - 15GB VRAM)
Google Colab provides free access to a Tesla T4 GPU, subject to dynamic session limits.

1. **Create a Notebook**: Go to [Google Colab](https://colab.research.google.com/), create a new Notebook, and navigate to **Runtime** -> **Change runtime type** -> select **T4 GPU**.
2. **Clone & Pull in a Cell**:
   ```python
   !git clone https://github.com/m-MARI444/Persistent-Sparse-Semantic-Architecture-PSSA-.git
   %cd Persistent-Sparse-Semantic-Architecture-PSSA-/pssa_project
   !git lfs install
   !git lfs pull
   ```
3. **Install Dependencies**:
   ```python
   !pip install transformers datasets psutil bitsandbytes truststore httpx urllib3
   ```
4. **Run Training**:
   ```python
   !python training/run_scaling_campaign.py --dataset=fineweb
   ```

---

## 🔄 Resuming the Training Campaign

To resume training from the last saved step (Step 25k), run the campaign runner script:

```bash
python3 training/run_scaling_campaign.py --dataset=fineweb
```

### Options:
* `--dataset=fineweb`: Default mode. Trains on the HuggingFace `fineweb-edu` sample dataset.
* `--dataset=hotpotqa`: Alternate mode. Trains on the `hotpot_qa` dataset.

### Verification:
Upon execution, the script will automatically detect `checkpoints/pssa_campaign_latest.pth` and display the following log:

```text
Initialize ~50M Parameter PSSA Model...
▶️ Loaded step=25000 phase=3
Resumed at Step 25000
[CKPT] Gradient checkpointing=OFF (Optimized)
```

---

## 🎨 Launching the Cognitive Dashboard

To inspect the 25k-trained model's internal slot mappings, timeline transitions, and causal matrices, start the visual web server:

```bash
python3 eval/visual_dashboard.py
```

### Accessing the Dashboard:
1. Open your browser and navigate to: **`http://localhost:8000/`** (or `8001` if port 8000 is occupied).
2. Enter custom text (e.g., *"John bought a red bicycle yesterday."*) to visualize PSSA-GPT's state arbitration winners, timeline routing, and attribute binding step-by-step!
