import os
import re
import gc
import sys
import warnings
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

# Increase MAX_TEXT_CHUNK to prevent Decompressed data too large errors from malformed image metadata
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# Suppress noisy warnings from transformers and other libraries
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoProcessor, AutoModelForImageTextToText

# Add parent directory to path to allow importing models/configs if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# Page Configuration & Styles
# ==============================================================================
st.set_page_config(
    page_title="ReasonFlow-VLA Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styled header and glassmorphic card containers
st.markdown("""
<style>
    /* Styling headers and custom font styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre, [class*="mono"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main-header {
        background: linear-gradient(135deg, #FF4B4B 0%, #7E57C2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #8A9Aad;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .card {
        background-color: rgba(33, 37, 43, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .model-badge {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    
    .badge-base {
        background-color: rgba(33, 150, 243, 0.15);
        color: #2196F3;
        border: 1px solid rgba(33, 150, 243, 0.3);
    }
    
    .badge-sft {
        background-color: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .badge-teacher {
        background-color: rgba(156, 39, 176, 0.15);
        color: #E040FB;
        border: 1px solid rgba(156, 39, 176, 0.3);
    }
    
    .thinking-box {
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #7E57C2;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    
    .answer-box {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #FF4B4B;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .telemetry-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Model Manager
# ==============================================================================

def clear_gpu_memory():
    """Unload variables, collect garbage, and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class VRAMSafeModelManager:
    """Manages active models, allowing multiple models in VRAM concurrently."""
    def __init__(self):
        if "models" not in st.session_state:
            st.session_state.models = {}
        if "processors" not in st.session_state:
            st.session_state.processors = {}

    def load_model(self, model_type: str, base_path: str, sft_path: str, teacher_lora_path: str):
        """Loads the requested model, applying PEFT LoRA if needed. Caches in VRAM."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        target_base = base_path if model_type == "Base LLM" else sft_path
        target_lora = teacher_lora_path if model_type == "Teacher Checkpoint" else None
        cache_key = f"{model_type}_{target_base}_{target_lora}"
        
        if cache_key in st.session_state.models:
            return st.session_state.models[cache_key], st.session_state.processors[cache_key]
        
        with st.spinner(f"Loading {model_type} ({target_base})..."):
            try:
                processor = AutoProcessor.from_pretrained(target_base)
                
                model = AutoModelForImageTextToText.from_pretrained(
                    target_base,
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    device_map="auto",
                    attn_implementation = "flash_attention_2"
                )
                
                if target_lora:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, target_lora)
                
                model.eval()
                
                if processor.tokenizer.pad_token_id is None:
                    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
                processor.tokenizer.padding_side = "left"
                
                st.session_state.models[cache_key] = model
                st.session_state.processors[cache_key] = processor
                
                st.toast(f"Successfully loaded {model_type}!", icon="✅")
                
            except Exception as e:
                st.error(f"Error loading model: {e}")
                raise e
                
        return st.session_state.models[cache_key], st.session_state.processors[cache_key]

    def unload_current(self):
        """Unload ALL active models completely and free VRAM."""
        try:
            for key in list(st.session_state.models.keys()):
                del st.session_state.models[key]
                del st.session_state.processors[key]
        except AttributeError:
            pass
        
        st.session_state.models = {}
        st.session_state.processors = {}
        
        clear_gpu_memory()
        gc.collect()
        clear_gpu_memory()

# Instantiate global manager
model_manager = VRAMSafeModelManager()

# ==============================================================================
# Helper Functions: Dataset loading, Waypoint Parsing, and Plotting
# ==============================================================================

@st.cache_data(show_spinner=True)
def fetch_stage2_dataset(dataset_name: str = "shreethar/FYP-Stage2-dataset", split: str = "test"):
    """Loads and caches the Hugging Face Stage 2 evaluation dataset."""
    from datasets import load_dataset
    try:
        return load_dataset(dataset_name, split=split)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

def parse_waypoints_and_thinking(text: str):
    """
    Parses thinking trace and coordinates from model generation text.
    Handles both standard bracket formats and Stage 2 `<ans>` formats.
    """
    thinking = ""
    answer = text
    
    # Extract text inside <think>...</think>
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
    elif "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].strip()
        answer = parts[1].strip()
    elif "<think>" in text:
        thinking = text.split("<think>", 1)[-1].strip()
        answer = ""
        
    # Clean answer wrapper tags
    clean_ans = answer.replace("<ans>", "").replace("</ans>", "").replace("<|im_end|>", "").strip()
    
    # Coordinate extraction regexes
    waypoints = None
    
    # 1. Try bracket format: [[x1, y1], [x2, y2], ...] or [x1, y1] [x2, y2]
    bracket_match = re.findall(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', clean_ans)
    if len(bracket_match) > 0:
        try:
            waypoints = [[float(x), float(y)] for x, y in bracket_match]
        except Exception:
            pass

    # 2. Try semicolon separated format: x1,y1;x2,y2;...
    if not waypoints:
        semicolon_match = re.search(r'([\d.]+,[\d.]+(?:;[\d.]+,[\d.]+)*)', clean_ans)
        if semicolon_match:
            pairs = semicolon_match.group(1).split(";")
            try:
                pts = []
                for p in pairs:
                    x, y = p.strip().split(",")
                    pts.append([float(x), float(y)])
                if len(pts) > 0:
                    waypoints = pts
            except Exception:
                pass
                
    # Normalize coordinates from 0-1000 scale to [0.0, 1.0] if necessary
    if waypoints:
        arr = np.array(waypoints, dtype=np.float32)
        if arr.max() > 2.0:
            arr = arr / 1000.0
        arr = np.clip(arr, 0.0, 1.0)
        waypoints = arr.tolist()
        
    # 3. Try bounding box format: [xmin, ymin, xmax, ymax]
    bbox = None
    if not waypoints:
        bbox_match = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', clean_ans)
        if bbox_match:
            try:
                bx1, by1, bx2, by2 = map(float, bbox_match.groups())
                if max(bx1, by1, bx2, by2) > 2.0:
                    bx1, by1, bx2, by2 = bx1/1000.0, by1/1000.0, bx2/1000.0, by2/1000.0
                bbox = [np.clip(v, 0.0, 1.0) for v in (bx1, by1, bx2, by2)]
            except Exception:
                pass

    return thinking, clean_ans, waypoints, bbox

def overlay_trajectory(image: Image.Image, waypoints: list, color: str = "#FF4B4B"):
    """Plots 2D waypoint paths and indices onto the PIL Image."""
    if image is None or not waypoints:
        return image
        
    img_draw = image.copy()
    w, h = img_draw.size
    draw = ImageDraw.Draw(img_draw)
    
    # Scale points to image width/height
    scaled_pts = [(int(x * w), int(y * h)) for x, y in waypoints]
    
    # Draw path line
    if len(scaled_pts) > 1:
        draw.line(scaled_pts, fill=color, width=4)
        
    # Draw circles at waypoints
    for idx, (px, py) in enumerate(scaled_pts):
        r = 6
        # Draw double ring for readability
        draw.ellipse([px-r-1, py-r-1, px+r+1, py+r+1], fill="black")
        draw.ellipse([px-r, py-r, px+r, py+r], fill="#2196F3", outline="white", width=2)
        
        # Add index numbers
        draw.text((px+8, py-10), str(idx+1), fill="yellow")
        
    return img_draw

def overlay_bboxes(image: Image.Image, bbox: list, color: str = "#4CAF50"):
    """Plots 2D bounding box onto the PIL Image."""
    if image is None or not bbox:
        return image
        
    img_draw = image.copy()
    w, h = img_draw.size
    draw = ImageDraw.Draw(img_draw)
    
    xmin, ymin, xmax, ymax = bbox
    px1, py1 = int(xmin * w), int(ymin * h)
    px2, py2 = int(xmax * w), int(ymax * h)
    
    draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
    return img_draw

# ==============================================================================
# UI Structure: Sidebar
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=64)
    st.markdown("<h2 style='margin-top: 0;'>Settings</h2>", unsafe_allow_html=True)
    
    # 1. Model configs
    st.subheader("Model Definitions")
    base_model_path = st.text_input("Base LLM Path", value="unsloth/Qwen3.5-4B")
    sft_model_path = st.text_input("Post SFT Model Path", value="shreethar/stage1_unsloth")
    
    # Automatically scan checkpoints folder for easy Teacher LoRA path selection
    local_checkpoints = []
    base_checkpoints_dir = "train/stage2/checkpoints/stage2_decoupled_mini"
    if os.path.isdir(base_checkpoints_dir):
        for step_dir in sorted(os.listdir(base_checkpoints_dir)):
            full_step_dir = os.path.join(base_checkpoints_dir, step_dir)
            if os.path.isdir(full_step_dir):
                lora_path = os.path.join(full_step_dir, "teacher_lora")
                if os.path.isdir(lora_path):
                    local_checkpoints.append(lora_path)
                    
    default_lora = local_checkpoints[-1] if local_checkpoints else "train/stage2/checkpoints/stage2_decoupled_mini/step_000100/teacher_lora"
    teacher_lora_path = st.text_input("Teacher LoRA Path", value=default_lora)
    
    # Model Loading Controls
    st.markdown("---")
    st.subheader("Model Status")
    
    # Show active model state
    active_models = list(st.session_state.models.keys()) if hasattr(st.session_state, "models") else []
    if active_models:
        st.markdown(f"🟢 **Active Models**: `{len(active_models)} loaded`")
        if st.button("Unload Models (Free VRAM)", use_container_width=True):
            model_manager.unload_current()
            st.rerun()
    else:
        st.markdown("🔴 **Active Models**: `None (Lazy Load)`")
        
    # Telemetry widget
    st.markdown("---")
    st.subheader("Telemetry")
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024**3)
        vram_res = torch.cuda.memory_reserved() / (1024**3)
        st.markdown(f"GPU VRAM Allocated: <span class='telemetry-val'>{vram_alloc:.2f} GB</span>", unsafe_allow_html=True)
        st.markdown(f"GPU VRAM Reserved: `{vram_res:.2f} GB`", unsafe_allow_html=True)
    else:
        st.markdown("GPU: `Not Available` (Running on CPU)")
        
    # Inference parameters
    st.markdown("---")
    st.subheader("Inference Settings")
    enable_thinking = st.toggle("Enable Thinking Mode", value=True)
    temp = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.05)
    max_tokens = st.slider("Max New Tokens", min_value=16, max_value=4096, value=128, step=16)
    rep_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.05)

# ==============================================================================
# UI Structure: Main Panel
# ==============================================================================

st.markdown("<div class='main-header'>ReasonFlow-VLA</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Interactive Inference & Model Comparison Dashboard</div>", unsafe_allow_html=True)

# Create layout tabs
tab1, tab2 = st.tabs(["📂 Dataset Comparison", "🎮 Custom Playground"])

# --- TAB 1: DATASET COMPARISON ---
with tab1:
    st.subheader("Select Dataset Sample")
    
    col_split, col_ds = st.columns(2)
    with col_split:
        selected_split = st.selectbox("Dataset Split", ["test", "train"])
    
    # Load dataset
    dataset = fetch_stage2_dataset(split=selected_split)
    
    if dataset is not None:
        # Exclude unwanted datasets
        excluded_datasets = {"pixmoama", "pixmocapqa", "pixmocap"}
        dataset = dataset.filter(lambda x: x["dataset"] not in excluded_datasets)
        
        # Group examples by their dataset source
        ds_names = sorted(list(set(dataset["dataset"])))
        with col_ds:
            selected_ds = st.selectbox("Dataset Source", ["All"] + ds_names)
        
        # Filter samples
        filtered_ds = dataset
        if selected_ds != "All":
            filtered_ds = dataset.filter(lambda x: x["dataset"] == selected_ds)
            
        # Example index selector
        num_examples = len(filtered_ds)
        example_idx = st.number_input("Example Index", min_value=0, max_value=num_examples-1, value=0)
        
        # Extract selected row
        sample = filtered_ds[example_idx]
        
        # Display sample information
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.markdown(f"**Sample ID**: `{sample.get('id', 'N/A')}`")
            st.markdown(f"**Dataset Source**: `{sample.get('dataset', 'N/A')}`")
            st.markdown(f"**Task Type**: `{sample.get('type', 'trajectory')}`")
        with col_meta2:
            st.markdown(f"**Original Instruction**:")
            st.info(sample["human"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display image / video observations
        frames = sample["frames"]  # list of PIL Images
        obs_images = frames if len(frames) > 1 else [frames[0]] # Pristine images for inference
        
        # Extract ground truth for visualization
        gt_text = sample.get("assistant", "")
        _, _, gt_waypoints, gt_bbox = parse_waypoints_and_thinking(gt_text)
        
        disp_images = []
        for frame in frames:
            disp_frame = frame.copy()
            if gt_waypoints:
                disp_frame = overlay_trajectory(disp_frame, gt_waypoints, color="#2196F3") # Blue for GT
            if gt_bbox:
                disp_frame = overlay_bboxes(disp_frame, gt_bbox, color="#2196F3") # Blue for GT
            disp_images.append(disp_frame)
            
        st.markdown("### Visual Observations (with Ground Truth)")
        
        if len(disp_images) == 1:
            # Single Image
            st.image(disp_images[0], caption="Observation Frame", width=448)
        else:
            # Video Frame Scrubber
            st.markdown(f"🎬 *Video Sequence detected ({len(disp_images)} frames)*")
            frame_idx = st.slider("Scrub Video Frames", 0, len(disp_images)-1, 0)
            st.image(disp_images[frame_idx], caption=f"Frame {frame_idx + 1} / {len(disp_images)}", width=448)
            
        st.markdown("**Ground Truth Output:**")
        st.info(gt_text)
            
        # Run comparison button
        st.markdown("---")
        st.markdown("### Model Comparison Output")
        
        st.markdown("**Select Models to Run:**")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            run_base = st.checkbox("Base LLM", value=False)
        with col_m2:
            run_sft = st.checkbox("Post SFT Model", value=False)
        with col_m3:
            run_teacher = st.checkbox("Teacher Checkpoint", value=True)
            
        models_to_test = []
        if run_base: models_to_test.append("Base LLM")
        if run_sft: models_to_test.append("Post SFT Model")
        if run_teacher: models_to_test.append("Teacher Checkpoint")
        
        if st.button("Run Selected Models", type="primary", use_container_width=True):
            if not models_to_test:
                st.warning("Please select at least one model to run.")
            else:
                cols = st.columns(len(models_to_test))
                
                # Setup prompt and images (same formatting as evaluation)
            # Reformat system prompt based on type
            is_traj = sample.get("type", "trajectory") == "trajectory"
            if is_traj:
                # Stage 2 Trajectory Prompt
                STAGE2_TRAJ_SYSTEM = (
                    "You are a robot manipulation assistant. Given an observation image and a task instruction, "
                    "predict the end-effector's 2D trajectory as 5 distinct waypoints showing the continuous movement. "
                    "Finally, output the coordinate list exactly once in this exact format: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]"
                )
                prompt_text = f"{STAGE2_TRAJ_SYSTEM}\n\nTask: {sample['human']}"
            else:
                # Stage 2 QA Prompt
                STAGE2_QA_SYSTEM = (
                    "You are a robot manipulation assistant. Answer questions about robot tasks, object affordances, "
                    "spatial relationships, and manipulation strategies based on the provided image or video frame. "
                    "If reasoning, think step-by-step. Finally, output the answer after </think>."
                )
                prompt_text = f"{STAGE2_QA_SYSTEM}\n\n{sample['human']}"
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video" if len(obs_images) > 1 else "image", 
                         "video" if len(obs_images) > 1 else "image": obs_images},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            for col_idx, model_name in enumerate(models_to_test):
                with cols[col_idx]:
                    st.markdown(f"<div class='model-badge badge-{model_name.lower().replace(' ', '-')}' style='font-size: 1.1rem;'>{model_name}</div>", unsafe_allow_html=True)
                    
                    try:
                        # 1. Load specific model in VRAM
                        model, processor = model_manager.load_model(
                            model_type=model_name,
                            base_path=base_model_path,
                            sft_path=sft_model_path,
                            teacher_lora_path=teacher_lora_path
                        )
                        
                        # 2. Formulate generation prompt
                        formatted = processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking
                        )
                        
                        inputs = processor(
                            text=[formatted],
                            images=obs_images if len(obs_images) == 1 else None,
                            videos=obs_images if len(obs_images) > 1 else None,
                            return_tensors="pt"
                        )
                        
                        # Move inputs to correct device
                        for k, v in inputs.items():
                            if isinstance(v, torch.Tensor):
                                inputs[k] = v.to(model.device)
                                
                        input_ids_len = inputs['input_ids'].shape[1]
                        
                        from transformers import TextStreamer
                        from threading import Thread
                        
                        class StreamlitStreamer(TextStreamer):
                            def __init__(self, tokenizer, show_thinking=True, **kwargs):
                                super().__init__(tokenizer, **kwargs)
                                self.text = ""
                                self.placeholder = st.empty()
                                self.in_think = False
                                self.show_thinking = show_thinking
                                self.expander = None
                                self.expander_placeholder = None
                                
                            def on_finalized_text(self, text: str, stream_end: bool = False):
                                self.text += text
                                
                                # Render streaming logic
                                if "<think>" in self.text and not self.in_think:
                                    self.in_think = True
                                    if self.show_thinking:
                                        self.expander = st.expander("💭 Reasoning Trace", expanded=True)
                                        self.expander_placeholder = self.expander.empty()
                                    
                                if self.in_think:
                                    if "</think>" in self.text:
                                        parts = self.text.split("</think>")
                                        think_text = parts[0].replace("<think>", "").strip()
                                        ans_text = parts[1].replace("<|im_end|>", "").strip()
                                        
                                        if self.show_thinking and self.expander_placeholder:
                                            self.expander_placeholder.markdown(f"<div class='thinking-box'>{think_text}</div>", unsafe_allow_html=True)
                                        self.placeholder.markdown(f"<div class='answer-box'>{ans_text}▌</div>", unsafe_allow_html=True)
                                    else:
                                        curr_think = self.text.replace("<think>", "").strip()
                                        if self.show_thinking and self.expander_placeholder:
                                            self.expander_placeholder.markdown(f"<div class='thinking-box'>{curr_think}▌</div>", unsafe_allow_html=True)
                                else:
                                    ans_text = self.text.replace("<|im_end|>", "").strip()
                                    self.placeholder.markdown(f"<div class='answer-box'>{ans_text}▌</div>", unsafe_allow_html=True)

                                if stream_end:
                                    ans_text = self.text.split("</think>")[-1].replace("<|im_end|>", "").strip()
                                    self.placeholder.markdown(f"<div class='answer-box'>{ans_text}</div>", unsafe_allow_html=True)
                                    
                        # 3. Generate tokens with streaming
                        streamer = StreamlitStreamer(processor.tokenizer, show_thinking=enable_thinking, skip_special_tokens=True)
                        
                        generation_kwargs = dict(
                            **inputs,
                            max_new_tokens=max_tokens,
                            stop_strings=["<|im_end|>"],
                            eos_token_id=processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                            repetition_penalty=rep_penalty,
                            tokenizer=processor.tokenizer,
                            do_sample=True,
                            temperature=temp,
                            streamer=streamer
                        )
                        
                        from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx
                        thread = Thread(target=model.generate, kwargs=generation_kwargs)
                        add_script_run_ctx(thread)
                        thread.start()
                        thread.join()
                        
                        # 4. Extract reasoning and waypoints after generation
                        pred_raw = streamer.text
                        thinking_text, answer_text, waypoints, bbox = parse_waypoints_and_thinking(pred_raw)
                        
                        # Plot trajectory if predicted
                        if waypoints:
                            st.markdown("🗺️ **Overlay Trajectory**:")
                            # Take the last frame for trajectory overlays
                            plotted_img = overlay_trajectory(obs_images[-1], waypoints)
                            st.image(plotted_img, caption="Predicted trajectory waypoints (1-5)")
                            
                        # Plot bounding box if predicted
                        if bbox:
                            st.markdown("🎯 **Overlay Bounding Box**:")
                            plotted_img = overlay_bboxes(obs_images[-1], bbox)
                            st.image(plotted_img, caption="Predicted bounding box")
                            
                    except Exception as gen_err:
                        st.error(f"Generation error: {gen_err}")
                        
            st.toast("Comparison finished!", icon="✅")

# --- TAB 2: CUSTOM PLAYGROUND ---
with tab2:
    st.subheader("Generalization Playground")
    st.markdown("Upload your own observations and test the models with custom prompts.")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        uploaded_files = st.file_uploader("Upload Image or Video Frames (Max 6 frames)", 
                                         type=["png", "jpg", "jpeg"], 
                                         accept_multiple_files=True)
        custom_prompt = st.text_area("Task Instruction / QA Prompt", 
                                     value="current goal is: Pick up the red object. Q: immediate next step?")
        
    with col_input2:
        custom_images = []
        if uploaded_files:
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                custom_images.append(img)
                
            if len(custom_images) == 1:
                st.image(custom_images[0], caption="Uploaded Image", width=350)
            else:
                st.markdown(f"🎬 *Video Sequence ({len(custom_images)} frames)*")
                cust_scrub = st.slider("Scrub Uploaded Frames", 0, len(custom_images)-1, 0)
                st.image(custom_images[cust_scrub], caption=f"Frame {cust_scrub+1}", width=350)
        else:
            st.info("Please upload an image to start testing.")
            
    if st.button("Generate Answer", type="primary", disabled=not (uploaded_files and custom_prompt), use_container_width=True):
        # Choose model to run playground on
        selected_play_model = st.selectbox("Select Model to Run", ["Teacher Checkpoint", "Post SFT Model", "Base LLM"])
        
        # Run Custom Inference
        try:
            model, processor = model_manager.load_model(
                model_type=selected_play_model,
                base_path=base_model_path,
                sft_path=sft_model_path,
                teacher_lora_path=teacher_lora_path
            )
            
            # Format custom template
            custom_system = (
                "You are a robot manipulation assistant. Given visual observations and a task prompt, "
                "reason step-by-step and predict waypoints or answer questions. "
                "Finally, output the answer after </think>."
            )
            prompt_text = f"{custom_system}\n\nTask: {custom_prompt}"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video" if len(custom_images) > 1 else "image", 
                         "video" if len(custom_images) > 1 else "image": custom_images},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            formatted = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            
            inputs = processor(
                text=[formatted],
                images=custom_images if len(custom_images) == 1 else None,
                videos=custom_images if len(custom_images) > 1 else None,
                return_tensors="pt"
            )
            
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)
                    
            input_ids_len = inputs['input_ids'].shape[1]
            
            from transformers import TextStreamer
            from threading import Thread
            
            st.markdown("### Playground Results")
            
            class StreamlitStreamerPlayground(TextStreamer):
                def __init__(self, tokenizer, show_thinking=True, **kwargs):
                    super().__init__(tokenizer, **kwargs)
                    self.text = ""
                    self.placeholder = st.empty()
                    self.in_think = False
                    self.show_thinking = show_thinking
                    self.expander = None
                    self.expander_placeholder = None
                    
                def on_finalized_text(self, text: str, stream_end: bool = False):
                    self.text += text
                    
                    if "<think>" in self.text and not self.in_think:
                        self.in_think = True
                        if self.show_thinking:
                            self.expander = st.expander("💭 Reasoning Trace", expanded=True)
                            self.expander_placeholder = self.expander.empty()
                        
                    if self.in_think:
                        if "</think>" in self.text:
                            parts = self.text.split("</think>")
                            think_text = parts[0].replace("<think>", "").strip()
                            ans_text = parts[1].replace("<|im_end|>", "").strip()
                            
                            if self.show_thinking and self.expander_placeholder:
                                self.expander_placeholder.markdown(f"<div class='thinking-box'>{think_text}</div>", unsafe_allow_html=True)
                            self.placeholder.markdown(f"<div class='answer-box'>{ans_text}▌</div>", unsafe_allow_html=True)
                        else:
                            curr_think = self.text.replace("<think>", "").strip()
                            if self.show_thinking and self.expander_placeholder:
                                self.expander_placeholder.markdown(f"<div class='thinking-box'>{curr_think}▌</div>", unsafe_allow_html=True)
                    else:
                        ans_text = self.text.replace("<|im_end|>", "").strip()
                        self.placeholder.markdown(f"<div class='answer-box'>{ans_text}▌</div>", unsafe_allow_html=True)

                    if stream_end:
                        ans_text = self.text.split("</think>")[-1].replace("<|im_end|>", "").strip()
                        self.placeholder.markdown(f"<div class='answer-box'>{ans_text}</div>", unsafe_allow_html=True)
                        
            streamer = StreamlitStreamerPlayground(processor.tokenizer, show_thinking=enable_thinking, skip_special_tokens=True)
            
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                stop_strings=["<|im_end|>"],
                eos_token_id=processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                repetition_penalty=rep_penalty,
                tokenizer=processor.tokenizer,
                do_sample=True,
                temperature=temp,
                streamer=streamer
            )
            
            with st.spinner("Generating output..."):
                from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                add_script_run_ctx(thread)
                thread.start()
                thread.join()
                
            pred_raw = streamer.text
            thinking_text, answer_text, waypoints, bbox = parse_waypoints_and_thinking(pred_raw)
            
            if waypoints:
                st.markdown("🗺️ **Overlay Trajectory**:")
                plotted_img = overlay_trajectory(custom_images[-1], waypoints)
                st.image(plotted_img, caption="Predicted trajectory waypoints")

            if bbox:
                st.markdown("🎯 **Overlay Bounding Box**:")
                plotted_img = overlay_bboxes(custom_images[-1], bbox)
                st.image(plotted_img, caption="Predicted bounding box")
                
        except Exception as play_err:
            st.error(f"Playground error: {play_err}")
