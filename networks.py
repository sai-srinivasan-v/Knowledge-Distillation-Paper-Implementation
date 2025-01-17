import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel,self).__init__()
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True 
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_input),
            nn.Linear(28*28,1200),
            nn.ReLU(),
            nn.Dropout(self.dropout_hidden),
            nn.Linear(1200,1200),
            nn.ReLU(),
            nn.Dropout(self.dropout_hidden),
            nn.Linear(1200,10)
        )

    def forward(self,x):
        return self.model(x)
    

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel,self).__init__()
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True 
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_input),
            nn.Linear(28*28,400),
            nn.ReLU(),
            nn.Dropout(self.dropout_hidden),
            nn.Linear(400,10)
        )
    def forward(self,x):
        return self.model(x)


class StudentModelSmall(nn.Module):
    def __init__(self):
        super(StudentModel,self).__init__()
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True 
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_input),
            nn.Linear(28*28,30),
            nn.ReLU(),
            nn.Dropout(self.dropout_hidden),
            nn.Linear(30,10)
        )
    def forward(self,x):
        return self.model(x)