import torch
import torchvision as tv
import torchvision.transforms as transform
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import SGD,Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F


''' Contains helper function for the 
reproducibility of the paper implementation: Distilling 
the Knowledge in a Neural Network
'''


def get_shape(dataset):
    return dataset[0][0].shape

def visualise(dataset):
    n = torch.randint(0, len(dataset), (1,)).item()
    rand_img = dataset[n]
    plt.imshow(rand_img[0].permute(1,2,0)) #Permute Converts CxHxW to HxWxC
    plt.title(rand_img[1])
    plt.axis('off')


def get_GPU():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def train(model,dataset,hparams,epoch,device):
    model.dropout_input = hparams['dropout_input']
    model.dropout_hidden = hparams['dropout_hidden']
    model = model.to(device)
    optimizer = SGD(model.parameters(),lr=hparams['lr'], momentum=hparams['momentum'], weight_decay=hparams['weight_decay'])
    lr_scheduler = StepLR(optimizer,gamma=hparams['lr_decay'], step_size=1)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    running_loss = 0.0
    correct = 0
    tot_samples = len(dataset.dataset)
    for batch, data in enumerate(dataset):
        X, y = data
        X,y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred_prob = model(X)
        loss = loss_fn(pred_prob, y)
        loss.backward()
        optimizer.step()


        running_loss += loss.item() * X.size(0)
        preds = torch.argmax(pred_prob,dim=1)
        correct += torch.sum(preds == y).item()
        if batch % 10 == 0:
            print(f'Epoch: {epoch+1}, training loss: {loss.item()}')
    avg_loss = running_loss / tot_samples
    epoch_acc = (correct / tot_samples) * 100
    print(f"Epoch {epoch+1} Completed: Avg Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    return avg_loss, epoch_acc


def evaluate_model(model,dataset,device,epoch = 0,loss_fn=nn.CrossEntropyLoss()):
    model.eval()
    running_loss,correct_preds = 0.0, 0
    total_samples = len(dataset.dataset)
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataset):
            X,y = X.to(device),y.to(device)
            pred_prob = model(X)
            loss = loss_fn(pred_prob,y)
            running_loss += loss.item()

            preds = torch.argmax(pred_prob,dim=1)
            correct_preds += torch.sum(preds == y).item()

    avg_loss = running_loss/total_samples
    acc = (correct_preds/total_samples) * 100
    print(f'Epoch: {epoch+1}, AVG Loss: {avg_loss}, accuracy per epoch: {acc}')
    return avg_loss, acc

def trainWithDistillation(student_model,
                          teacher_model, 
                          dataset,
                          device, 
                          hparams,epochs):

    student_model.dropout_input = hparams['dropout_input']
    student_model.dropout_hidden = hparams['dropout_hidden']
    
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  

    optimizer = SGD(student_model.parameters(), 
                    lr=hparams['lr'], 
                    momentum=hparams['momentum'], 
                    weight_decay=hparams['weight_decay'])
    lr_scheduler = StepLR(optimizer, gamma=hparams['lr_decay'], step_size=1)


    def distillation_loss(teacher_pred, student_pred, y, T, alpha):
        """
        Loss = alpha * (distillation loss) + (1 - alpha) * (cross-entropy loss)
        """
        soft_targets = F.kl_div(
            F.log_softmax(student_pred / T, dim=1), 
            F.softmax(teacher_pred / T, dim=1), 
            reduction='batchmean'
        ) * (T ** 2) * alpha
        hard_targets = F.cross_entropy(student_pred, y) * (1 - alpha)
        return soft_targets + hard_targets

    student_model.train()
    running_loss = 0.0
    correct = 0
    tot_samples = len(dataset.dataset)
    
    for batch_idx, (X, y) in enumerate(dataset):
        X, y = X.to(device), y.to(device)


        with torch.no_grad():
            teacher_pred = teacher_model(X) 
        
        student_pred = student_model(X) 
        
        loss = distillation_loss(
            teacher_pred=teacher_pred, 
            student_pred=student_pred, 
            y=y, 
            T=hparams['T'], 
            alpha=hparams['alpha']
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = torch.argmax(student_pred, dim=1)
        correct += torch.sum(preds == y).item()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epochs+1}, training loss: {loss.item()}')
    

    lr_scheduler.step()


    avg_loss = running_loss / tot_samples
    epoch_acc = (correct / tot_samples) * 100
    print(f"Epoch {epochs+1} Completed: Avg Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    return avg_loss, epoch_acc


def hparamToString(hparam):
    hparam_str = ''
    for k, v in sorted(hparam.items()):
        hparam_str += k + '=' + str(v)+ ', '
    return hparam_str

def hparamDictToTuple(hparam):
    hparam_tuple = [v for k,v in hparam.items()]
    return tuple(hparam_tuple)



