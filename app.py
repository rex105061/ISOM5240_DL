import streamlit as st
import os
import torch

# ===== 内存优化和稳定性配置 =====
# 禁用多进程，避免 semaphore 泄漏
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 强制使用 CPU（GPU 可能反而有问题）
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 设置 torch 的内存优化
torch.set_num_threads(1)

from transformers import pipeline
from gtts import gTTS
import tempfile
from PIL import Image

st.set_page_config(page_title="Fairy Tale Storyteller", page_icon="🧚")
st.header("✨ Turn Your Image into a Fairy Tale ✨")
st.write("For kids 3-10 years old - upload a picture and get a magical story!")

# ===== 模型加载函数（使用更小的模型 + device_map） =====
@st.cache_resource
def load_caption_model():
    try:
        # 使用更小的配置，强制 CPU
        return pipeline(
            "image-to-text", 
            model="Salesforce/blip-image-captioning-base",
            device=-1,  # -1 表示使用 CPU
            model_kwargs={"low_cpu_mem_usage": True}
        )
    except Exception as e:
        st.error(f"Failed to load caption model: {e}")
        raise

@st.cache_resource
def load_story_model():
    try:
        return pipeline(
            "text-generation",
            model="roneneldan/TinyStories-1M",  # 最小的版本
            max_new_tokens=120,  # 减少生成长度
            do_sample=True,
            temperature=0.7,
            device=-1,  # 强制 CPU
            model_kwargs={"low_cpu_mem_usage": True}
        )
    except Exception as e:
        st.error(f"Failed to load story model: {e}")
        raise

def enforce_word_limit(story, min_words=50, max_words=100):
    words = story.split()
    if len(words) > max_words:
        truncated = ' '.join(words[:max_words])
        last_period = truncated.rfind('.')
        if last_period > 0:
            truncated = truncated[:last_period+1]
        return truncated
    return story

def img2text(image_path, captioner):
    image = Image.open(image_path)
    # 压缩图片以减少内存
    image.thumbnail((512, 512))
    result = captioner(image)
    return result[0]["generated_text"]

def caption_to_story(description, story_model):
    prompt = f"Once upon a time, {description.lower()} "
    result = story_model(
        prompt, 
        max_new_tokens=120,  # 明确指定，避免使用默认值
        do_sample=True,
        temperature=0.7
    )
    full_story = result[0]['generated_text']
    story = full_story.replace(prompt, "").strip()
    # 如果故事太短，添加一个后缀
    if len(story.split()) < 50:
        story += " They lived happily ever after. The end."
    story = enforce_word_limit(story, 50, 100)
    return story

def story_to_audio(story):
    tts = gTTS(text=story, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# ===== 主界面 =====
uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 保存上传的图片
    temp_image_path = f"temp_{uploaded_file.name}"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    st.image(uploaded_file, caption="Your picture", width=400)
    
    # 用按钮触发生成，避免自动加载模型
    if st.button("✨ Generate Story & Audio ✨", type="primary"):
        try:
            # Step 1: 图片理解
            with st.spinner("🔍 Understanding your picture..."):
                captioner = load_caption_model()
                scenario = img2text(temp_image_path, captioner)
            st.caption(f"📝 I see: *{scenario}*")
            
            # Step 2: 生成故事
            with st.spinner("📖 Writing your fairy tale..."):
                story_model = load_story_model()
                story = caption_to_story(scenario, story_model)
            
            st.subheader("🧚 Your Fairy Tale")
            st.write(story)
            
            # Step 3: 字数统计
            word_count = len(story.split())
            if 50 <= word_count <= 100:
                st.success(f"📊 Word count: {word_count} ✅ (Goal: 50-100)")
            else:
                st.warning(f"📊 Word count: {word_count} (Ideal range: 50-100)")
            
            # Step 4: 生成音频
            with st.spinner("🔊 Creating audio..."):
                audio_path = story_to_audio(story)
            
            # Play 按钮
            st.audio(audio_path)
            
            # 清理临时文件
            os.unlink(audio_path)
            
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
            st.info("Try refreshing the page or using a different image.")
    
    # 清理图片文件
    if os.path.exists(temp_image_path):
        os.unlink(temp_image_path)
