import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
import os

# 配置页面
st.set_page_config(page_title="Fairy Tale Storyteller", page_icon="🧚")
st.header("✨ Turn Your Image into a Fairy Tale ✨")
st.write("For kids 3-10 years old - upload a picture and get a magical story!")

# 加载模型（使用缓存避免重复加载）
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    return pipeline(
        "text-generation",
        model="roneneldan/TinyStories-33M",  # 专为儿童故事设计
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7
    )

# 字数控制函数
def enforce_word_limit(story, max_words=100):
    words = story.split()
    if len(words) > max_words:
        truncated = ' '.join(words[:max_words])
        last_period = truncated.rfind('.')
        if last_period > 0:
            truncated = truncated[:last_period+1]
        return truncated
    return story

# 图片到描述
def img2text(image_path, captioner):
    result = captioner(image_path)
    return result[0]["generated_text"]

# 描述到故事
def caption_to_story(description, story_model):
    prompt = f"Once upon a time, {description.lower()} "
    result = story_model(prompt)
    full_story = result[0]['generated_text']
    # 清理：去掉 prompt 部分
    story = full_story.replace(prompt, "").strip()
    story = enforce_word_limit(story, 100)
    return story

# 故事到语音
def story_to_audio(story):
    tts = gTTS(text=story, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# 主界面
uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 保存上传的图片
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    st.image(uploaded_file, caption="Your picture", use_column_width=True)
    
    # 加载模型
    with st.spinner("🖼️ Understanding your picture..."):
        captioner = load_caption_model()
        scenario = img2text(uploaded_file.name, captioner)
    st.caption(f"📝 I see: *{scenario}*")
    
    # 生成故事
    with st.spinner("📖 Writing a fairy tale for you..."):
        story_model = load_story_model()
        story = caption_to_story(scenario, story_model)
    
    st.subheader("🧚 Your Fairy Tale")
    st.write(story)
    
    # 显示字数统计（调试/教学用）
    word_count = len(story.split())
    st.caption(f"📊 Word count: {word_count} (50-100 words goal)")
    
    # 生成语音
    with st.spinner("🔊 Creating audio..."):
        audio_path = story_to_audio(story)
    
    # 播放按钮
    if st.button("🔊 Read my story aloud!"):
        st.audio(audio_path)
    
    # 清理临时文件
    os.unlink(audio_path)
    os.unlink(uploaded_file.name)
