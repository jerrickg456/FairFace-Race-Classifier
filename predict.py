import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define race labels (adjust as needed)
race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']

# Load model
device = torch.device('cpu')
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=None)
model.fc = nn.Linear(model.fc.in_features, len(race_labels))

# Load trained weights
model.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt', map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        race = race_labels[predicted.item()]
    
    return race
