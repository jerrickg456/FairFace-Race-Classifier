import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

def predict_image(img_path):
    device = torch.device("cpu")  # FORCE CPU

    model_fair_7 = torchvision.models.resnet34(pretrained=False)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)  # 18 outputs: 7 race + 9 age + 2 gender

    model_fair_7.load_state_dict(torch.load(
        'fair_face_models/res34_fair_align_multi_7_20190809.pt',
        map_location=torch.device('cpu')
    ))
    model_fair_7.eval()

    # Load image with OpenCV
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return "Error loading image!"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces_rects) == 0:
        return "No face detected"

    # Just take first face
    (x, y, w, h) = faces_rects[0]
    face_crop = img_rgb[y:y+h, x:x+w]

    # Resize to 224x224
    face_resized = cv2.resize(face_crop, (224, 224))

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_image = trans(face_resized).unsqueeze(0)

    with torch.no_grad():
        outputs = model_fair_7(input_image)

        race_outputs = outputs[:, :7]  # First 7 = Race

        probs = torch.nn.functional.softmax(race_outputs[0], dim=0)
        predicted_class = torch.argmax(probs).item()

    race_names = [
        "White",
        "Black",
        "Latino_Hispanic",
        "East Asian",
        "Southeast Asian",
        "Indian",
        "Middle Eastern"
    ]

    return race_names[predicted_class]
