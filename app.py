import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

classes = ["NORMAL", "PNEUMONIA"]

def predict(image):

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()

    return f"{classes[pred]} ({confidence*100:.1f}% confidence)"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction"),
    title="Chest X-ray Pneumonia Detection",
    description="Upload a Chest X-ray image.\n\n⚠️ This is an educational demo, not a substitute for professional medical diagnosis."
)

demo.launch()