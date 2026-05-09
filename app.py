import streamlit as st
from transformers import pipeline
import scipy.io.wavfile
import io

# --- Model Caching ---
@st.cache_resource
def load_image_to_text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generation_model():
    # Using a model specifically for children's stories and relatively lightweight
    # roneneldan/TinyStories-33M is a good candidate.
    # We need to specify trust_remote_code=True for some models like TinyStories.
    return pipeline("text-generation", model="roneneldan/TinyStories-33M", trust_remote_code=True)

@st.cache_resource
def load_text_to_audio_model():
    # Using a lightweight TTS model suitable for Streamlit Cloud
    return pipeline("text-to-audio", model="facebook/mms-tts-eng")

# --- Function Part ---
def img2text(url):
    image_to_text_model = load_image_to_text_model()
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def generate_story(scenario):
    story_pipe = load_story_generation_model()
    
    # Crafting a prompt for a fairy tale for young children
    prompt = f"Once upon a time, there was a {scenario}. This is a short fairy tale for kids aged 3-10: "
    
    # Generate story with word count control
    # max_new_tokens controls the length of the generated text
    # Adjust max_new_tokens to target 50-100 words (approx 70-140 tokens)
    story_results = story_pipe(prompt, max_new_tokens=100, num_return_sequences=1, truncation=True)
    story = story_results[0]["generated_text"]
    
    # Post-process to ensure word count and clean up prompt repetition
    story = story.replace(prompt, "").strip()
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]) + "..."
    elif len(words) < 50:
        # For simplicity, if too short, we'll just use it. 
        # In a real app, you might regenerate or add a message.
        pass
        
    return story

def text_to_audio(text):
    audio_pipe = load_text_to_audio_model()
    audio_data = audio_pipe(text)
    
    # Convert numpy array to WAV byte stream
    audio_array = audio_data["audio"]
    sample_rate = audio_data["sampling_rate"]
    
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate=sample_rate, data=audio_array)
    buffer.seek(0) # Rewind the buffer to the beginning
    return buffer.getvalue()

# --- Main Part ---
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
st.header("Turn Your Image to Audio Story")

uploaded_file = st.file_uploader("Select an Image...")

if uploaded_file is not None:
    # Save file locally (Streamlit handles this temporarily)
    # For pipeline, we can directly pass the BytesIO object or file path
    # For simplicity, we'll save it to a temporary file for img2text, 
    # as some image-to-text models might expect a file path.
    bytes_data = uploaded_file.getvalue()
    file_path = f"./temp_image_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stage 1: Image to Text
    st.text('Processing img2text...')
    scenario = img2text(file_path)
    st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story
    st.text('Generating a story...')
    story = generate_story(scenario)
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio
    st.text('Generating audio data...')
    audio_bytes = text_to_audio(story)

    # Play button
    st.audio(audio_bytes, format='audio/wav')
    st.success("Audio generated successfully!")
