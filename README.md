
# AllLinLog: All contextual information utilized with the Linear self-attention encoder in Log-based anomaly detection

This repository contains the implementation of **AllLinLog**, a framework for efficient log-based anomaly detection.
The project explores the use of linear self-attention for large-scale log analysis, applied to datasets 
such as **BGL** and **HDFS**.

Before using this project, you can verify that the repository works correctly by following these steps:

---

## Quick Start
### 1. Clone the repository
```bash
git clone https://github.com/tw13023/AllLinLog.git
cd AllLinLog
```

### 2. Software Requirements:
1. `torch==2.7.1+cu128`
2. `numpy==2.3.1`
3. `pandas==2.3.1`
4. `scikit-learn==1.7.0`
5. `tqdm==4.67.1`
6. `tiktoken==0.9.0`
7. `linformer==0.2.3`
8. `psutil==7.0.0`
9. `matplotlib==3.10.3`

### 3. Reconstructing Large LogsLarge Logs

The original logs are too large for GitHub.
They have been compressed and split into smaller chunks.

### Reconstruct BGL.log
```bash
cd /logs
gunzip BGL.log.gz
```

### Reconstruct HDFS.log
```bash
cat logs/HDFS_part_*.gz | gunzip > HDFS.log
```
### 4. Run the Jupyter Notebook

- `allinlog_BGL_inMem_GPT4BPE.ipynb` for **BGL** dataset
- `allinlog_HDFS_inMEM_GPT4BPE.ipynb` for **HDFS** dataseat

