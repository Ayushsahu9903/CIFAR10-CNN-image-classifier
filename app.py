import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import CNN

# ---------------- Page config ----------------
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🧠",
    layout="centered"
)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

EMOJIS = {
    'airplane': '✈️', 'automobile': '🚗', 'bird': '🐦', 'cat': '🐱', 'deer': '🦌',
    'dog': '🐶', 'frog': '🐸', 'horse': '🐴', 'ship': '🚢', 'truck': '🚚'
}

DEVICE = torch.device("cpu")

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #6a11cb15, #2575fc15);
        text-align: center;
        border: 1px solid #2575fc40;
    }
    .prediction-text {
        font-size: 1.8rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("cifar10_cnn_model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# Must match training transform exactly
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------- Header ----------------
st.markdown('<p class="main-title">🧠 CIFAR-10 Image Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let the CNN guess what it is</p>', unsafe_allow_html=True)

# Model stats
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Test Accuracy", "76.0%")
with col_b:
    st.metric("Classes", "10")
with col_c:
    st.metric("Architecture", "CNN")

with st.expander("ℹ️ About this model"):
    st.write("""
    This is a Convolutional Neural Network trained from scratch on the **CIFAR-10** dataset,
    achieving **86% accuracy** on the held-out test set.
    It can classify images into 10 categories:
    """)
    st.write(", ".join([f"{EMOJIS[c]} {c}" for c in CLASS_NAMES]))

st.divider()

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.image(image, caption="Your image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    conf_value = confidence.item() * 100

    with col2:
        st.markdown(f"""
            <div class="result-box">
                <div style="font-size:3rem;">{EMOJIS[predicted_class]}</div>
                <div class="prediction-text">{predicted_class.upper()}</div>
                <div style="color:#666;">Confidence: {conf_value:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
        st.progress(min(int(conf_value), 100))

    st.divider()
    st.subheader("📊 Class probabilities")
    prob_dict = {f"{EMOJIS[CLASS_NAMES[i]]} {CLASS_NAMES[i]}": float(probs[0][i]) for i in range(10)}
    st.bar_chart(prob_dict)

else:
    st.info("👆 Upload an image to get a prediction")

st.markdown("---")
st.caption("Built with PyTorch + Streamlit | CNN trained on CIFAR-10")
