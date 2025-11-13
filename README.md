# 🚀 Next-Word Prediction, Regularization Experiments & Deep Learning on MNIST  
A comprehensive deep-learning project involving MLP-based next-word prediction, model regularization experiments on synthetic datasets, and CNN/MLP comparisons on MNIST.

---

## 👥 Team Members
- **Sai Gawali**  
- **Siddharth Nayak**  
- **Siddhesh Patil**

---

## 📌 Project Overview

This project combines **Natural Language Processing**, **Optimization**, and **Deep Learning** through the following components:

- MLP-based next-word text generation  
- Embedding learning & visualization  
- Streamlit deployment of a text generator  
- Regularization experiments on the Moons dataset  
- MLP vs CNN benchmarking on MNIST  
- Transfer learning using pretrained CNNs  

The project emphasizes deep understanding of representation learning, generalization, and visualization of high-dimensional ML behaviors.

---

# 🧠 1. Next-Word Prediction using MLP

### ✔️ Core Pipeline
- Built a **next-word prediction model** using a feed-forward neural network (MLP)
- Trained on **two datasets**: one natural-language dataset & one structured-text dataset
- Implemented vocabulary construction, preprocessing, dataset creation, and sliding-window context extraction

### ✔️ Model Architecture
- Embedding layer (32/64 dimensions)  
- 1–2 hidden layers of 1024 neurons  
- ReLU/Tanh activation  
- Softmax output over vocabulary  

### ✔️ Highlights
- Trained for 500–1000 epochs  
- Tracked training & validation loss  
- Generated example predictions and analyzed learning patterns  
- Compared word frequencies, token distributions, and vocabulary size  
- Created embedding visualizations using t-SNE  
- Observed semantic grouping patterns (verbs, nouns, pronouns, synonyms, code-tokens, etc.)

---

# 🌐 1.4 Streamlit Next-Word Generator

A fully interactive **Streamlit application** was built with:
- Adjustable context length  
- Embedding dimensions  
- Activation function choice  
- Random seed  
- Temperature (controls randomness)  

The app predicts:
- Next **k words**  
- Next **line** (structured text datasets)  

Handled out-of-vocabulary words gracefully and included multiple model variants for user selection.

visite website: [https://predictnext.streamlit.app/](https://predictnext.streamlit.app/)
---

# 📝 1.5 Comparative Analysis (Natural vs Structured Text)

Compared:
- Dataset size & vocabulary richness  
- Predictability of context  
- Loss curves & model performance  
- Embedding cluster structures (semantic vs syntactic)  

Derived insights about natural language smoothness vs structured data rigidity.

---

# 🌙 2. Moons Dataset & Regularization

Built the **Make-Moons** dataset *without scikit-learn*. Added noise levels 0.1, 0.2, 0.3 for robustness evaluation.

### ✔️ Models Trained
1. **MLP (Early Stopping)**  
2. **MLP with L1 Regularization**  
3. **MLP with L2 Regularization**  
4. **Logistic Regression with Polynomial Features**

### ✔️ Analysis Performed
- Validation AUROC vs L1 penalty λ  
- Sparsity of layer weights with L1  
- Smoothness/margin effects of L2  
- Class imbalance experiments (70:30)  
- Robustness scores across different noise levels  
- Decision boundary visualizations for all 4 models  
- Comparative table for:
  - Accuracy  
  - AUROC  
  - Model parameters  
  - Noise robustness  

---

# 🔢 3. MNIST & CNN Experiments

### ✔️ 3.1 MLP on MNIST
- Built an MLP with layers: **30 → 20 → 10**  
- Compared performance with:
  - Logistic Regression  
  - Random Forest  
- Evaluated:
  - Accuracy  
  - F1-score  
  - Confusion matrix  
- Visualized t-SNE embeddings for trained vs untrained models  
- Tested MNIST-trained model on **Fashion-MNIST** to see domain mismatch  
- Compared t-SNE embeddings for MNIST vs Fashion-MNIST

---

### ✔️ 3.2 CNN on MNIST
- Built a custom CNN:
  - Conv(32 filters, 3×3)
  - MaxPool  
  - Dense(128)  
  - Output(10)  
- Trained end-to-end on MNIST  
- Compared performance with two pretrained CNNs (e.g., AlexNet, MobileNet)  
- Benchmarked:
  - Accuracy  
  - F1-score  
  - Confusion matrix  
  - Parameter count  
  - Inference time  

---

# 📂 Repository Structure
```
│── next_word_mlp/
│── embeddings_visualization/
│── streamlit_app/
│── moons_regularization/
│── mnist_mlp/
│── mnist_cnn/
│── question1.ipynb
│── question2.ipynb
│── question3.ipynb
│── README.md
```

---

# 🛠️ Tech Stack
- Python  
- NumPy  
- PyTorch  
- Matplotlib  
- Streamlit  
- t-SNE (sklearn)  

---

# 🎯 Summary

This project demonstrates:
- Skill in sequence modeling without RNNs or transformers  
- Understanding of embeddings and semantic structure  
- Ability to deploy ML models as interactive applications  
- Deep knowledge of regularization techniques and robustness  
- Hands-on experience with MLPs & CNNs for computer vision  
- Familiarity with transfer learning and embedding visualization  

---
