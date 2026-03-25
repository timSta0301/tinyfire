import torch
from PIL import Image
from torchvision import transforms
from classifier import TinyFireCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "first_tests/basic_cnn/tinyfirecnn_best.pth"

model = TinyFireCNN().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

img_path = "dataset/classifier_ds/test/sate2.jpeg"   # path to test image
img = Image.open(img_path).convert("RGB")
x = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logit = model(x)
    prob_no_fire = torch.sigmoid(logit).item()
    pred = "no fire" if prob_no_fire > 0.5 else "fire"
    prob_fire = 1.0 - prob_no_fire

print("Probability no fire:", prob_no_fire)
print("Probability fire:", prob_fire)
print("Prediction:", pred)