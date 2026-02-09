# HiMAP: Tracking-Free Motion Forecasting with History Reconstruction from Occupancy Maps

**HiMAP** is a robust motion forecasting framework designed for scenarios where multi-object tracking (MOT) is unstable or unavailable.  
Instead of relying on track IDs or tracker intermediates (e.g., affinity matrices), HiMAP reconstructs agent history directly from **spatiotemporal historical occupancy maps** built from unordered detections, and predicts multi-modal future trajectories with a query-based decoder.

---

### Pipeline
![](<img width="2317" height="698" alt="pipeline" src="https://github.com/user-attachments/assets/2922dbe4-7201-4304-aaa2-d7dc8878c755" />)

---


## News
- **[31.01.2026]** HiMAP is accepted by ICRA 2026.

---

## Getting Started

**Step 1**: Clone

```
git clone https://github.com/XuYiMing83/HiMAP.git
cd HiMAP
```

**Step 2**: create a conda environment and install the dependencies:

```
conda env create -f environment.yml
conda activate HiMAP
```

**Step 3**: install the [Argoverse 2 API](https://github.com/argoverse/av2-api) and download the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html) following the [Argoverse 2 User Guide](https://argoverse.github.io/user-guide/getting_started.html).

---

## Training & Evaluation
```
python train_net.py --root /path/to/dataset_root/ --processed HiMAP --train_batch_size 8 --val_batch_size 8 --test_batch_size 8 --devices 4
```

---

### Testing
```
python test.py --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
```


