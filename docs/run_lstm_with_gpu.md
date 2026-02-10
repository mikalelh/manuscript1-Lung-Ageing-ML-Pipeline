# Running the LSTM model with GPU (hospital server)

Follow these steps on your hospital server.

---

## Step 1: Check if a GPU is available

In a terminal:

```bash
nvidia-smi
```

- If you see a table with GPU name, driver version, and memory → you have a GPU.
- If you get "command not found" or "no devices" → the machine has no GPU or drivers aren’t installed; you’ll need to ask IT or use the cloud GPU instance instead.

---

## Step 2: Activate your environment and check PyTorch + CUDA

```bash
cd ~/ipython_notebooks/manuscript1_ageing_inthelung/msl_aging_pipeline
conda activate scanpy_env
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

- If it prints **CUDA available: True** and a GPU name → you can go to Step 4 and run the script.
- If it prints **CUDA available: False** → PyTorch is CPU-only; do Step 3 to install a CUDA build.

---

## Step 3: Install PyTorch with CUDA (only if Step 2 said False)

First check which CUDA version your driver supports:

```bash
nvidia-smi
```

Look at the top right for "CUDA Version: X.X". Then install PyTorch with matching CUDA.

**Option A – Conda (recommended if you use conda):**

```bash
# CUDA 11.8 (common on hospital clusters)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Or CUDA 12.1
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Option B – Pip:**

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then run the check again:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Step 4: Run the LSTM script

From the pipeline directory, with the same environment active:

```bash
cd ~/ipython_notebooks/manuscript1_ageing_inthelung/msl_aging_pipeline
conda activate scanpy_env
python models/lstm_aging_model.py
```

At the start you should see something like:

```text
Device: cuda
Using 1407 genes for LSTM input.
```

If you see **Device: cpu** instead, PyTorch still isn’t using the GPU; re-check Step 2 and 3.

---

## Step 5: (Optional) Run in the background

If you want to disconnect and let it run:

```bash
nohup python models/lstm_aging_model.py > lstm_log.txt 2>&1 &
tail -f lstm_log.txt
```

`tail -f` shows the log live; Ctrl+C only stops tail, not the job.

---

## Quick reference

| Step | Command / check |
|------|------------------|
| 1. GPU on machine | `nvidia-smi` |
| 2. PyTorch sees GPU | `python -c "import torch; print(torch.cuda.is_available())"` |
| 3. Install CUDA PyTorch | `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia` (or pip, see above) |
| 4. Run LSTM | `python models/lstm_aging_model.py` |

If your hospital server has no GPU or you can’t install CUDA PyTorch there, run the same steps on the cloud instance (RTX 5090) you showed earlier; the walkthrough is the same.
