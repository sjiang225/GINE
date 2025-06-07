
# ESPNet: Edge-Aware Graph Representation Learning over Analyst–Firm Bipartite Networks for Earnings Surprise Prediction
This repository provides the official implementation of our ICDM 2025 submission: ESPNet.  
ESPNet is a GINE-based graph neural network framework designed to predict earnings surprises by modeling analyst–firm interactions as a heterogeneous bipartite graph. The framework integrates edge-aware message passing and relational feature propagation to capture complex financial signals for robust prediction and downstream portfolio analysis.

![Model Structure](./model.png)

## 📂 Files

- `data_sim.R` – R script to simulate synthetic scRNA-seq data.
- `run_fssc.py` – Main Python script to run FSSC clustering.
- `run.sh` – Shell script to execute the clustering demo.


## 🛠️ Dependencies

The code has been tested with:

- Python 3.8  
- PyTorch 1.12   
- Python packages: `numpy`, `pandas`, `scikit-learn`, `torch`  

