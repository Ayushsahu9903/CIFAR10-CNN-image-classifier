
## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Ayushsahu9903/CIFAR10-CNN-image-classifier.git
cd CIFAR10-CNN-image-classifier
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## ☁️ Deployment

This app is deployed on [Streamlit Community Cloud](https://share.streamlit.io).

To deploy your own copy:
1. Fork this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select this repo, branch `main`, and main file `app.py`
4. Click **Deploy**

## 🛠️ Tech Stack

- **PyTorch** – model training & inference
- **Torchvision** – dataset handling and image transforms
- **Streamlit** – web app framework
- **Pillow** – image processing

## 📊 Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32×32 color images across 10 classes (50,000 training / 10,000 test).

## 📸 Demo

Upload an image → get instant prediction with confidence score and full class probability chart.

## 🙋‍♂️ Author

**Ayush Sahu**
GitHub: [@Ayushsahu9903](https://github.com/Ayushsahu9903)

Feel free to connect or raise an issue if you spot a bug!
