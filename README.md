# Knowledge Distillation for Neural Network

This project implements and explores knowledge distillation techniques, as described in the seminal paper "Distilling the Knowledge in a Neural Network". The aim is to train a 
smaller, more efficient student model to mimic the behavior of a larger, more complex teacher model, preserving performance while reducing resource consumption.

## Project Structure 

funcs.py: Contains core helper functions for the project.

networks.py: Contains the teacher and student models implemented for the project.

teacher.ipynb: Jupyter Notebook for training the teacher model.

student.ipynb: Jupyter Notebook for training the student model using knowledge distillation.


## How it Works

Knowledge Distillation
Knowledge distillation is a technique where a teacher model trains a student model by transferring its learned knowledge. The student model learns not only from the true labels but also from the soft probabilities (soft targets) produced by the teacher model.
This process is governed by a temperature parameter T and a weighting factor alpha to balance the contribution of distillation loss and true label loss.

# Results and Observations
The temperature parameter (T) significantly influenced the student model's performance. Higher temperatures resulted in smoother probability distributions, aiding knowledge transfer.
The Student model was observed to have improved in accuracy post-distillation compared to the models accuracy pre-distillation. 

## Acknowledgments

This project is inspired by the paper "Distilling the Knowledge in a Neural Network" by Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.

For questions or contributions, feel free to contact me or submit a pull request!

