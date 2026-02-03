import gradio as gr
from PIL import Image
import requests
import io
import numpy as np

# --- LOGIC SECTION ---
def predict_drawing(image):
    """
    This function takes the image from the sketchpad, processes it,
    sends it to the API, and returns the prediction.
    """

    # 1. Handle Gradio input format (Gradio sometimes returns a dict)
    if isinstance(image, dict) and 'composite' in image:
        image = image['composite']

    # 2. Convert Numpy array to PIL Image
    if hasattr(image, 'astype'):
        image = Image.fromarray(image.astype('uint8'))

    # 3. Ensure grayscale
    try:
        if image.mode != 'L':
            image = image.convert('L')
    except Exception as e:
        print(f"Error converting image to grayscale: {e}")
        return "Error processing image"

    # 4. Invert colors
    # Sketchpad draws black on white (255 background, 0 ink).
    # Neural networks (like MNIST/QuickDraw) usually train on white ink on black background.
    image = Image.eval(image, lambda x: 255 - x)

    # 5. Save image to bytes to send via API
    img_binary = io.BytesIO()
    image.save(img_binary, format='PNG')
    img_binary = img_binary.getvalue()

    # TODO: Define your API URL
    # Since we are running locally, it should look like [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)
    api_url = "http://api:5075/predict"

    # TODO: Send the POST request
    # Use requests.post()
    # Pass img_binary as data
    # Warning: The API expects a file upload, so you might need to check how to send raw bytes 
    # or wrap it in a 'files' dictionary depending on your API implementation.
    # For this tutorial, we assume the API handles raw bytes or multipart/form-data.
    try:
        # Hint: response = requests.post(url, files={"file": ...})
        response = requests.post(api_url, files={"file": ("drawing.png", img_binary, "image/png")})
    except requests.exceptions.ConnectionError:
        return "Error: API is down. Is api.py running?"

    # TODO: Parse the response
    if response.ok:
        # Extract the JSON content
        # Return the 'class' or 'label' from the JSON
        return response.json().get("class", "Unknown")
    else:
        print(f"Request failed: {response.status_code} - {response.reason}")
        return "Prediction failed"


# --- UI SECTION (Don't touch this!) ---
if __name__=='__main__':
    # We define the interface
    interface = gr.Interface(
        fn=predict_drawing, 
        inputs="sketchpad", 
        outputs='label',
        live=False, # Set to True if you want real-time feedback (can be slow)
        title="Quick, Draw! Pictionary",
        description="Draw a cat, an airplane, or a donut. Click 'Submit' to see if the model recognizes it!",
    )

    # Launch the server
    # server_name='0.0.0.0' allows access from other machines if needed
    interface.launch(debug=True, share=False, server_name='0.0.0.0', server_port=7860)