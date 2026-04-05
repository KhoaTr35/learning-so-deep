import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Phân loại ảnh Caltech-256", page_icon="📸")
st.title("Ứng dụng Phân loại Ảnh - Caltech-256 📸")

# --- TẠO MENU CHỌN MODEL ---
model_choice = st.selectbox(
    "Vui lòng chọn mô hình bạn muốn dùng để dự đoán:",
    ("ResNet50 (CNN)", "ViT-B_16 (Transformer)")
)

# --- HÀM TẢI RESNET50 ---
@st.cache_resource
def load_resnet():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 257)
    model.load_state_dict(torch.load('resnet50_best.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# --- HÀM TẢI ViT-B_16 ---
@st.cache_resource
def load_vit():
    model = models.vit_b_16(weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, 257)
    model.load_state_dict(torch.load('vit_b_16_best.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_categories():
    with open('categories.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]

# --- TẢI MODEL DỰA TRÊN LỰA CHỌN ---
if model_choice == "ResNet50 (CNN)":
    model = load_resnet()
else:
    model = load_vit()
    
categories = load_categories()

# --- XỬ LÝ ẢNH ---
uploaded_file = st.file_uploader("Chọn một bức ảnh để phân loại...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh bạn đã tải lên', use_column_width=True)
    
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    st.write(f"⏳ Đang phân tích bằng **{model_choice}**...")
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        
    st.subheader("Kết quả phân tích:")
    for i in range(3):
        class_name = categories[top3_catid[i].item()]
        confidence = top3_prob[i].item() * 100
        st.write(f"**{class_name}**: {confidence:.2f}%")
        st.progress(int(confidence))