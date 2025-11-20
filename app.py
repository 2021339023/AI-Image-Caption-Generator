import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️")

st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and let AI generate a creative caption for it!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        # Load BLIP model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Prepare inputs
        inputs = processor(images=image, return_tensors="pt")

        # Generate caption
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.success("Caption generated!")
    st.subheader("📝 Caption:")
    st.write(caption)
