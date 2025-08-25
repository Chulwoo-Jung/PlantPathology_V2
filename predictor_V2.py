import torch
import torch.nn.functional as F

class Predictor:
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict_proba(self, images):
        self.model.eval()

        with torch.no_grad():
            inputs = images.to(self.device)
            outputs = self.model(inputs)
            pred_proba = F.softmax(outputs, dim=1)

        return pred_proba
    
    def predict(self, images):
        pred_proba = self.predict_proba(images)
        pred_labels = torch.argmax(pred_proba, dim=1)
        return pred_labels