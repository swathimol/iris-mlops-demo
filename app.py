# app.py
import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(data)[0]
    iris_classes = ["Setosa", "Versicolor", "Virginica"]
    return f"🌸 Predicted Iris Flower: {iris_classes[pred]}"

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🌼 Iris Flower Prediction App",
    description="A simple demo using Logistic Regression and the Iris dataset."
)

demo.launch()
