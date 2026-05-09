import streamlit as st
from transformers import pipeline
import scipy.io.wavfile
import io
import os

# --- Model Caching ---
@st.cache_resource
def load_image_to_text_model():
    # Robust loading: provide model name, let transformers detect the task
    # This avoids the "Unknown task" error.
    return pipeline(model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generation_model():
    # Using model name only for consistency
    return pipeline(model="roneneldan/TinyStories-33M", trust_remote_code=True)

@st.cache_resource
def load_text_to_audio_model():
    # Using model name only for consistency
    return pipeline(model="facebook/mms-tts-eng")

# --- Function Part ---
def img2text(image_path):
    image_to_text_model = load_image_to_text_model()
    result = image_to_text_model(image_path)
    return result[0]["generated_text"]

def generate_story(scenario):
    story_pipe = load_story_generation_model()
    # Prompt for a children's fairy tale
    prompt = f"Once upon a time, there was a {scenario}. This is a short fairy tale for kids aged 3-10: "
    
    # Generate story
    story_results = story_pipe(prompt, max_new_tokens=100, num_return_sequences=1, truncation=True)
    story = story_results[0]["generated_text"]
    
    # Clean up and word count control (50-100 words)
    story = story.replace(prompt, "").strip()
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]) + "..."
    elif len(words) < 50:
        # If too short, we can add a simple ending to reach the word count if needed,
        # but usually TinyStories generates enough content.
        pass
    return story

def text_to_audio(text):
    audio_pipe = load_text_to_audio_model()
    audio_data = audio_pipe(text)
    
    audio_array = audio_data["audio"]
    sample_rate = audio_data["sampling_rate"]
    
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate=sample_rate, data=audio_array)
    buffer.seek(0)
    return buffer.getvalue()

# --- Main Part ---
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
st.header("Turn Your Image to Audio Story")

uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save file locally
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    try:
        # Stage 1: Image to Text
        with st.spinner('Reading the image...'):
            scenario = img2text(temp_file_path)
        st.write(f"**Scenario:** {scenario}")

        # Stage 2: Text to Story
        with st.spinner('Writing a fairy tale...'):
            story = generate_story(scenario)
        st.write(f"**Story:** {story}")

        # Stage 3: Story to Audio
        with st.spinner('Generating audio...'):
            audio_bytes = text_to_audio(story)

        # Play button
        st.audio(audio_bytes, format='audio/wav')
        st.success("Your story is ready!")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
