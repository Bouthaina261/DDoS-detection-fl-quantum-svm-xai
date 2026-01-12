# DDoS Attack Detection in SDN using Federated Learning (FL) — Quantum, XAI & SVM

This project aims to **detect Distributed Denial-of-Service (DDoS) attacks** in a  
**Software-Defined Networking (SDN)** environment using **Federated Learning (Flower)**,  
allowing model training **without centralizing sensitive network traffic data**.

Each federated client represents an **SDN controller or network domain** that locally observes its own traffic flows.

The project combines:
- **Federated Learning (FL)**
- **Classical Machine Learning (SVM)**
- **Quantum Machine Learning (Variational Quantum Circuits)**
- **Explainable AI (XAI) using SHAP**

---

## 1. Project Objectives

- Efficiently detect **DDoS attacks** in SDN networks  
- Preserve **data privacy** using Federated Learning  
- Compare:
  - a **classical approach (SVM)**
  - a **quantum approach (VQC)**
- Provide **model interpretability** through **XAI techniques**

---

## 2. Dataset
- The dataset used in this project is available on Kaggle:
- **Dataset**: CIC-DDoS2019  
- **File used**:  
  `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- **Link**: https://www.kaggle.com/datasets/ishasingh03/friday-workinghours-afternoon-ddos
- **Actual size**: **225,745 network flows**
- **Number of features**: 85
- **Data type**: network flow statistics

### Class Distribution (approximate)

- **BENIGN (Normal traffic)**: ~43%
- **DDoS / DoS (Attack traffic)**: ~57%

---

## 3. Selected Features (8)

Both approaches (SVM and Quantum) use **exactly the same features** to ensure a **fair comparison**:

- `Flow Duration`
- `Total Fwd Packets`
- `Total Backward Packets`
- `Flow IAT Mean`
- `Flow Bytes/s`
- `Flow Packets/s`
- `Fwd IAT Mean`
- `Bwd IAT Mean`

These features capture:
- traffic intensity,
- temporal behavior,
- bidirectional flow characteristics.

---

## 4. Federated Learning Architecture

- **FL framework**: Flower (simulation)
- **Number of clients**: `6`
- **Server strategy**: `FedAvg`
- **Evaluation**: centralized on the server only

### FL Workflow
1. The server initializes and sends the global model
2. Each client trains locally on its private data
3. Clients send model parameters to the server
4. The server aggregates parameters using **FedAvg**
5. The process repeats for multiple rounds

---

## 5. Approach 1 — FL + SVM (Linear SVM via SGDClassifier)

### Objective
Train a **linear SVM** in a federated manner to detect DDoS attacks using the **FedAvg strategy over 10 Federated Learning rounds**, where each client performs local training before server-side aggregation.

### Data Preprocessing
1. Data cleaning (`inf → NaN`)
2. Median imputation
3. Binary labeling:
   - `BENIGN → 0`
   - `DDoS / DoS → 1`
4. **Standardization** using `StandardScaler`
   - `fit` on global training set
   - `transform` on global test set (no data leakage)

### Client Data Partitioning
- Each client receives:
  - `1/6` of attack flows
  - `1/6` of benign flows
- The **original dataset distribution is preserved**

### Model
- `SGDClassifier`
- `loss="hinge"` → linear SVM
- Incremental learning via `partial_fit`

### Main Hyperparameters

| Parameter        | Value | Description |
|------------------|-------|-------------|
| `NUM_CLIENTS`    | 6     | Number of FL clients |
| `NUM_ROUNDS`     | 10    | Federated rounds |
| `LOCAL_EPOCHS`   | 3     | Local epochs per round |
| `BATCH_SIZE`     | 512   | Mini-batch size |
| `ALPHA`          | 1e-4  | L2 regularization |
| `ETA0`           | 1e-2  | Learning rate |

### Global Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Hinge loss

Metrics are **computed only on the server** using a **global test set**.

---

## 6. Approach 2 — FL + Quantum (Flower + PennyLane)

### Objective
Explore **Variational Quantum Classifiers (VQC)** for DDoS detection in a federated setting, using **FedAvg over 25 Federated Learning rounds**, where each client trains locally before server aggregation.

### Data Preprocessing
- Cleaning + `dropna`
- Binary labeling:
  - `1` if `"DoS"` or `"DDoS"`
  - `0` otherwise
- Normalization:
  - `MinMaxScaler → [0, π]`
  - Suitable for quantum rotations

### Quantum Model (VQC)
- Framework: **PennyLane**
- Encoding:
  - `RY(x_i)` applied to each qubit
- Variational layer:
  - `Rot(θ, φ, ω)`
- Output:
  - `⟨PauliZ⟩` → probability → class

### Training
- Quantum gradient descent (`qml.grad`)
- Local subsampling (quantum cost constraint)
- FedAvg aggregation of quantum weights

### Metrics
- Loss
- Accuracy
- F1-score
- Per-round curves

---

## 7. XAI — Explainable AI with SHAP (FL + SVM)

To ensure **model transparency and trust**, an **XAI layer** is added using **SHAP (SHapley Additive exPlanations)**.

### Why SHAP?
- The final model is **linear**
- `shap.LinearExplainer` is:
  - fast
  - theoretically grounded
- SHAP enables:
  - **global explanations**
  - **local explanations**

---

### SHAP Visualizations

1. **Global Feature Importance (Bar Plot)**
   - `mean(|SHAP|)`
   - Identifies the most influential features

2. **Beeswarm Plot (Impact & Direction)**
   - SHAP > 0 → pushes toward **DDoS**
   - SHAP < 0 → pushes toward **Benign**
   - Color indicates feature value (high / low)

3. **Waterfall Plot (Single Flow Explanation)**
   - Explains why a specific flow is classified as DDoS
   - Shows individual feature contributions

### Typical Observations
- `Flow Packets/s` is the dominant feature
- `Bwd IAT Mean` and `Flow Duration` capture key temporal patterns
- Results align with known DDoS behavior in the literature

---

## 8. Results

- Accuracy ≈ **88–91%**
- High precision
- Recall depends on decision threshold
- F1-score provides the most balanced evaluation

The **FL + SVM + SHAP** combination provides:
- strong detection performance
- full interpretability
- privacy preservation

---

## 9. Technologies Used

- Python
- Flower (Federated Learning)
- Scikit-learn
- PennyLane (Quantum Machine Learning)
- SHAP (Explainable AI)
- NumPy / Pandas / Matplotlib

---

## 10. Conclusion

This project demonstrates that:
- **Federated Learning** is well-suited for SDN security
- **Linear SVMs** remain highly effective for DDoS detection
- **Quantum models** are promising but computationally expensive
- **XAI is essential** for trustworthy security systems

---

## 11. Future Work

- Real SDN integration (ONOS / OpenDaylight)
- Advanced non-IID FL scenarios
- Adversarial attack resilience
- Hybrid SVM–Quantum models
- Edge / MEC deployment