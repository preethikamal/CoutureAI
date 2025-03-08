import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

# Set up the title and description
st.title("CoutureAI - Personalized Clothing Visualization")
st.write(
    "Welcome to CoutureAI! Describe your desired outfit, and we'll generate a realistic image for you. "
    "You can visualize how the clothing will look before making a purchase."
)

# Input for Hugging Face API token
api_token = st.text_input(
    "Enter your Hugging Face API Token:",
    type="password",
    value=""  # No pre-filled value for security reasons
)

# Initialize the Hugging Face Inference Client
if api_token:
    client = InferenceClient(provider="hf-inference", api_key=api_token)
else:
    st.warning("Please enter your Hugging Face API token to proceed.")

# Text input for outfit description
outfit_description = st.text_area(
    "Describe your desired outfit:",
    placeholder="e.g., A stylish red dress with floral patterns, suitable for summer"
)

# Additional inputs for customization
background_color = st.color_picker("Choose a background color:", "#FFFFFF")
style_preference = st.selectbox("Select a style:", ["Casual", "Formal", "Streetwear", "Vintage", "Bohemian"])
material = st.text_input("Preferred fabric or material (optional):", placeholder="e.g., Silk, Cotton, Denim")
accessories = st.text_input("Include accessories (optional):", placeholder="e.g., Handbag, Sunglasses, Necklace")

# Button to generate the image
if st.button("Generate Outfit"):
    if not api_token:
        st.error("Please enter a valid Hugging Face API token.")
    elif not outfit_description:
        st.warning("Please describe your desired outfit.")
    else:
        with st.spinner("Generating your outfit... This may take a few seconds."):
            try:
                # Construct the enhanced prompt
                prompt = (f"{outfit_description}, styled in {style_preference} fashion, "
                          f"made of {material if material else 'a suitable fabric'}, "
                          f"with accessories like {accessories if accessories else 'none'}. "
                          f"The background is {background_color}.")
                
                # Generate the image using the Hugging Face API
                image = client.text_to_image(
                    prompt,
                    model="stabilityai/stable-diffusion-xl-base-1.0"
                )

                # Display the generated image
                st.image(image, caption="Generated Outfit", use_column_width=True)

                # Option to download the image
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                st.download_button(
                    label="Download Image",
                    data=buffered.getvalue(),
                    file_name="generated_outfit.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
