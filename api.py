import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, jsonify

# Load the pre-trained model
model = tf.keras.models.load_model('yoga_pose_classifier.h5')  # Replace with your model path

# Define a function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess the image for the model.
    """
    img = image.load_img(image_path, target_size=target_size)  # Load image and resize
    img_array = image.img_to_array(img)                       # Convert to array
    img_array = np.expand_dims(img_array, axis=0)             # Add batch dimension
    img_array = img_array / 255.0                             # Normalize pixel values
    return img_array

# Define a function to make predictions
def classify_image(image_path):
    """
    Classify the image as 'Yoga Pose' or 'Not Yoga Pose'.
    """
    # Preprocess the image
    img = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(img)
    classes = ['Not Yoga Pose', 'Yoga Pose']  # Update with your class names

    # Get the index of the class with the highest probability
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    # Return the result
    return {
        'prediction': classes[predicted_class],
        'confidence': round(confidence, 2)
    }

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to upload an image and get the prediction.
    """
    # Check if an image file is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the uploaded image to a temporary path
        image_path = 'temp_image.png'
        file.save(image_path)

        # Classify the uploaded image
        result = classify_image(image_path)

        # Return the result in JSON format
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
