import os
import json
import torch
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import requests
from transformers import CLIPProcessor, CLIPModel
from gtts import gTTS
import tempfile

# Set up OpenRouter API Key
OPENROUTER_API_KEY = "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"  # Replace with actual key

# Load the CNN Model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}\app\trained_model\plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)


# Load Class Names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Load CLIP model for image verification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def is_plant_or_leaf(image):
    """Check if the image is a plant or leaf using CLIP."""
    text_labels = ["a leaf", "a plant", "a car", "a building", "an animal"]
    inputs = clip_processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    plant_score = probs[0][0] + probs[0][1]  # Sum of "leaf" and "plant" probabilities
    return plant_score.item() > 0.5  # Convert tensor to float

# Function to Load and Preprocess Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Disease
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to Get Treatment Recommendation using OpenRouter AI
def get_treatment_recommendation(disease_name):
    healthy_keywords = ["healthy", "no disease", "normal leaf", "healthy leaf", "no issues"]
    
    if any(keyword in disease_name.lower() for keyword in healthy_keywords):
        return "✅ The leaf is healthy. No treatment is needed. Keep monitoring for any future issues."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourwebsite.com",
        "X-Title": "Plant Disease Detector"
    }
    
    data = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are an expert in plant disease treatment. Provide up to 10 short, point-wise treatments with Hindi translations."},
            {"role": "user", "content": f"How do I treat {disease_name} in plants?"}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        
        if "choices" in response_json:
            treatment = response_json['choices'][0]['message']['content']
            return format_recommendation(treatment)
        else:
            return f"API Error: {response_json}"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to Format Recommendation
def format_recommendation(text):
    points = text.split("\n")[:10]  # Limit to 10 points
    formatted_points = []
    
    for point in points:
        point = point.strip()
        hindi_translation = translate_to_hindi(point)
        
        if hindi_translation:  # If translation exists, use it
            formatted_points.append(f"- {hindi_translation}")
        else:  # Otherwise, show the original English point
            formatted_points.append(f"- {point}")  
    
    return "\n".join(formatted_points)


# Function to Translate Text to Hindi
def translate_to_hindi(text):
    hindi_translation = {
        "Remove infected areas": "संक्रमित क्षेत्रों को हटा दें",
        "Improve air circulation": "हवा के प्रवाह में सुधार करें",
        "Use fungicides": "फफूंदनाशकों का उपयोग करें",
        "Avoid overhead watering": "ऊपर से पानी देने से बचें",
        "Maintain plant hygiene": "पौधों की सफाई बनाए रखें",
        "Apply copper-based products": "तांबा-आधारित उत्पादों का उपयोग करें",
        "Use resistant varieties": "प्रतिरोधी किस्मों का उपयोग करें",
        "Monitor for pests": "कीटों की निगरानी करें"
    }
    return hindi_translation.get(text, "")  # Return empty string if translation not available

# Function to Generate and Save Audio
def generate_audio(text):
    tts = gTTS(text=text, lang="hi")  # Hindi audio
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Streamlit App UI
st.title('🌿 AgriTech Solution: Leaf Disease Detector')

uploaded_image = st.file_uploader("📤 Upload a plant image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image")

    with col2:
        if st.button('🔍 Predict Disease'):
            if is_plant_or_leaf(image):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'🌱 **Prediction:** {str(prediction)}')

                # Fetch treatment recommendation from OpenRouter AI
                treatment = get_treatment_recommendation(prediction)
                st.info(f'💡 **Treatment Recommendation:**\n{treatment}')
                
                # Generate Audio Output
                audio_path = generate_audio(treatment)
                st.audio(audio_path, format='audio/mp3')
            else:
                st.error("❌ The uploaded image is not a plant or leaf. Please upload a valid image.")
