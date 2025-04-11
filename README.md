# 🌸 Streamlit Iris Classifier

A simple, interactive web app built with **Streamlit** that uses a **Random Forest Classifier** to predict the species of an Iris flower based on its sepal and petal measurements.

![Streamlit Iris Classifier Screenshot](https://user-images.githubusercontent.com/your-screenshot-url) <!-- (Optional: Replace with a real screenshot) -->

---

## 🔍 Features

- Interactive sliders to input flower measurements
- Real-time predictions using a trained machine learning model
- Clean and intuitive UI with Streamlit
- Powered by `scikit-learn` and the classic Iris dataset

---

## 📦 Technologies Used

- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas**

---

## 🚀 Run the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/streamlit-iris-classifier.git
cd streamlit-iris-classifier
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📊 How It Works

1. Loads the Iris dataset from `sklearn.datasets`
2. Trains a `RandomForestClassifier` on the flower features
3. Takes user input from the sidebar using Streamlit widgets
4. Predicts the species based on input
5. Displays the predicted species in real-time

---

## 🔧 Future Improvements

- Add prediction probability chart
- Support CSV uploads for bulk predictions
- Add model selection (SVM, KNN, etc.)
- Deploy on Streamlit Cloud or Render

---

## 🖼️ Sample Output

```plaintext
The predicted species is: Iris-virginica
```

---

## 🧠 Author

**Rudra Dubey**  

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
