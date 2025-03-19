# 📚 Distilling the Knowledge in a Neural Network  

🔍 **Paper Re-Implementation** | 🎯 **Knowledge Distillation for Efficient Models**  

This project implements and explores **Knowledge Distillation**, based on the seminal paper  
[*"Distilling the Knowledge in a Neural Network"*](https://arxiv.org/pdf/1503.02531) by **Geoffrey Hinton, Oriol Vinyals, and Jeff Dean**.  

The goal is to train a **smaller, more efficient Student model** that mimics the behavior of a **larger, more complex Teacher model**, preserving performance while reducing computational cost.  

---

## 📂 Project Structure  
📂 funcs.py – Core helper functions for training and evaluation.
📂 networks.py – Implementation of the Teacher and Student models.
📒 teacher.ipynb – Jupyter Notebook for training the Teacher Model.
📒 student.ipynb – Jupyter Notebook for training the Student Model using Knowledge Distillation.


---

## 💡 How Knowledge Distillation Works  

📌 **Knowledge Distillation** is a technique where a **large Teacher model** trains a **smaller Student model** by transferring knowledge. The student learns from:  
✔ **True labels** (hard targets)  
✔ **Soft probabilities** (soft targets) produced by the Teacher  

### 📏 Key Parameters  
- **Temperature (T)** – Controls the softness of the Teacher’s probability distribution.  
- **Alpha (α)** – Balances the contribution of **distillation loss** and **true label loss**.  

---

## 📊 Results & Key Observations  

✔ **Higher temperature (T) values led to smoother probability distributions**, improving knowledge transfer.  
✔ **The Student model’s accuracy improved post-distillation**, demonstrating the effectiveness of distillation.  
✔ **A well-trained Student model required significantly fewer resources than the Teacher while maintaining strong performance.**  

📉 **Training Graphs & Accuracy Comparisons** *(Add visual results here if possible!)*  

---

## 🚀 Getting Started  

### 1️⃣ Install Dependencies  
```bash
pip install torch torchvision numpy matplotlib
