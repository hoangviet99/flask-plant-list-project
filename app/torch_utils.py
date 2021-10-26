import torch
import torchvision
import os 
import io
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
from torch._C import device
from PIL import Image
from torchvision.transforms.transforms import Resize

# 1. Load model

MODEL_PATH_LEAF = './model/model_5_plant_leaf.pth'
MODEL_PATH_FLOWER = './model/model_5_plant_flower.pth'
MODEL_PATH_FRUIT = './model/model_5_plant_fruit.pth'
MODEL_PATH_OVERALL = './model/model_5_plant_overall.pth'

out_features = 5

def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out =self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}
    
    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, epochs, result):
        print("Epoch: [{}/{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch+1, epochs, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, out_features)
        
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self): #by freezing all the layers but the last one we allow it to warm up (the others are already good at training)
        for param in self.network.parameters():
            param.require_grad=False
        for param in self.network.fc.parameters():
            param.require_grad=True
            
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad=True

model = ResNet()

# 2. Img to tensor

def transform_image(image_bytes):
    preprocess = T.Compose([
        # T.Resize(224),
        # T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return preprocess(image).unsqueeze(0)

# 3. Get Prediction Image

def get_prediction(input_batch, type_predict):
    if type_predict == 'leaf':
        checkpoint = torch.load(MODEL_PATH_LEAF)
    if type_predict == 'flower':
        checkpoint = torch.load(MODEL_PATH_FLOWER)
    if type_predict == 'fruit':
        checkpoint = torch.load(MODEL_PATH_FRUIT)
    if type_predict == 'overall':
        checkpoint = torch.load(MODEL_PATH_OVERALL)

    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        output = model(input_batch)

    output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().data.numpy()

    prediction = int(torch.max(output.data, 1)[1].numpy())
    best_accuracy = probabilities[prediction]

    result = [best_accuracy, prediction]

    return result