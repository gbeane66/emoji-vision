import streamlit as st
import torch
import open_clip
from PIL import Image
import numpy as np

# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model,_,preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Function to predict descriptions and probabilities
def predict(image, descriptions):
    image = preprocess(image).unsqueeze(0).to(device)
    text = tokenizer(descriptions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # logits_per_image, logits_per_text = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        text_features /= text_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
        print(probs)
    
    return descriptions[probs.argmax()], probs.max()

# Streamlit app
def main():
    st.set_page_config(layout="wide")

    st.title("🔥How Well Does this Model Understand Images? Can we use Emoji's to describe them? Let's find out!")

    # Instructions for the user
    st.markdown("---")
    st.markdown("### Upload an image and let's see how well the model understands it!")

    # Upload image through Streamlit with a unique key
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"], key="uploaded_image")

    if uploaded_image is not None:
        # Convert the uploaded image to PIL Image
        pil_image = Image.open(uploaded_image)

        # Limit the height of the displayed image to 400px
        st.image(pil_image, caption="Uploaded Image.", use_container_width=True, width=200)
        
        # Instructions for the user
        st.markdown("### Truth or Fiction?")
        st.markdown("Describe the image (tip: you can use a single emoji!): One of the descriptions needs to be true.")

        # Get user input for descriptions
        description1 = st.text_input("Description 1:", placeholder='🍎')
        description2 = st.text_input("Description 2:", placeholder='🐎')
        description3 = st.text_input("Description 3:", placeholder='🏚️')

        descriptions = [description1, description2, description3]

        # Button to trigger prediction
        if st.button("Predict"):
            if all(descriptions):
                # Make predictions
                best_description, best_prob = predict(pil_image, descriptions)

                # Display the highest probability description and its probability
                st.write(f"**Best Description:** {best_description}")
                st.write(f"**Prediction Probability:** {best_prob:.2%}")

                # Display progress bar for the highest probability
                st.progress(float(best_prob))

if __name__ == "__main__":
    main()
