"""
Real-IAD Industrial Defect Detection & Localization Web Application
PatchCore-style anomaly detection powered by WideResNet-50-2 feature representations.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import wide_resnet50_2
import streamlit as st
from huggingface_hub import hf_hub_download

# ==============================================================================
# 1. Page Configuration & Custom Dark Industrial CSS Styling
# ==============================================================================
st.set_page_config(
    page_title="Real-IAD Industrial Defect Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

:root {
    --bg: #0E1113;
    --sidebar: #15191C;
    --surface: #1A1F22;
    --border: #2A3034;

    --text: #E5E7E9;
    --muted: #92999F;

    --accent: #C9A227;
    --accent-hover: #D8B33A;

    --normal: #4C9A78;
    --defective: #B85C5C;
}

/* Main application */
.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Streamlit Material Icons preservation */
[data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stSidebarCollapseButton"] span,
.material-symbols-rounded,
.material-icons {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}

/* Main title */
.main-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 2.15rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
    margin-bottom: 0.15rem;
}

/* Subtitle */
.sub-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--accent);
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
}

/* Horizontal separators */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.4rem 0;
}

/* Section headings */
h1, h2, h3 {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}

/* Normal status */
.status-normal {
    background: #17231D;
    color: #72B493;
    padding: 12px 14px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 1.25rem;
    text-align: center;
    border: 1px solid #315E48;
    margin-bottom: 1rem;
}

/* Defective status */
.status-defective {
    background: #281A1A;
    color: #D47A7A;
    padding: 12px 14px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 1.25rem;
    text-align: center;
    border: 1px solid #6B3838;
    margin-bottom: 1rem;
}

/* Primary button */
.stButton > button {
    background: var(--accent);
    color: #111315;
    border: 1px solid var(--accent);
    border-radius: 4px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background: var(--accent-hover);
    border-color: var(--accent-hover);
    color: #111315;
}

/* Selectbox and uploader */
div[data-baseweb="select"] > div {
    background-color: var(--surface);
    border-color: var(--border);
    border-radius: 4px;
}

[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
}

/* Metric values */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: var(--muted);
}

/* Sidebar markdown */
section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text);
}

/* Sidebar horizontal rules */
section[data-testid="stSidebar"] hr {
    border-top: 1px solid var(--border);
}

/* Code-style benchmark values */
code {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent);
    background: #1B2023;
    border-radius: 3px;
    padding: 2px 5px;
}

/* Image containers */
img {
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. Benchmark Results (Real-IAD Single-View Evaluation)
# ==============================================================================
BENCHMARK_DATA = {
    "audiojack": {"S-AUROC": 0.8288, "I-AUROC": 0.9944, "P-AUPRO": 0.9451, "Threshold": 21.4030},
    "bottle_cap": {"S-AUROC": 0.9618, "I-AUROC": 0.9966, "P-AUPRO": 0.8929, "Threshold": 18.1935},
    "button_battery": {"S-AUROC": 0.8479, "I-AUROC": 0.9947, "P-AUPRO": 0.9163, "Threshold": 20.8063},
    "end_cap": {"S-AUROC": 0.8329, "I-AUROC": 0.9883, "P-AUPRO": 0.8100, "Threshold": 19.3390},
    "eraser": {"S-AUROC": 0.9379, "I-AUROC": 0.9978, "P-AUPRO": 0.8465, "Threshold": 19.8173},
    "fire_hood": {"S-AUROC": 0.9596, "I-AUROC": 0.9962, "P-AUPRO": 0.8639, "Threshold": 22.4464},
    "mint": {"S-AUROC": 0.7268, "I-AUROC": 0.9857, "P-AUPRO": 0.7043, "Threshold": 18.7771},
    "mounts": {"S-AUROC": 0.9264, "I-AUROC": 0.9970, "P-AUPRO": 0.8771, "Threshold": 22.7317},
    "pcb": {"S-AUROC": 0.9023, "I-AUROC": 0.9974, "P-AUPRO": 0.8749, "Threshold": 21.3793},
    "phone_battery": {"S-AUROC": 0.9362, "I-AUROC": 0.9968, "P-AUPRO": 0.8633, "Threshold": 22.0916},
    "plastic_nut": {"S-AUROC": 0.8607, "I-AUROC": 0.9975, "P-AUPRO": 0.8821, "Threshold": 21.0721},
    "plastic_plug": {"S-AUROC": 0.9573, "I-AUROC": 0.9969, "P-AUPRO": 0.7802, "Threshold": 20.1564},
    "porcelain_doll": {"S-AUROC": 0.8674, "I-AUROC": 0.9799, "P-AUPRO": 0.8552, "Threshold": 18.3277},
    "regulator": {"S-AUROC": 0.8646, "I-AUROC": 0.9974, "P-AUPRO": 0.9166, "Threshold": 18.4501},
    "rolled_strip_base": {"S-AUROC": 0.9861, "I-AUROC": 0.9985, "P-AUPRO": 0.9765, "Threshold": 19.4087},
    "sim_card_set": {"S-AUROC": 0.9780, "I-AUROC": 0.9982, "P-AUPRO": 0.9557, "Threshold": 20.1280},
    "switch": {"S-AUROC": 0.9304, "I-AUROC": 0.9713, "P-AUPRO": 0.9171, "Threshold": 21.9950},
    "tape": {"S-AUROC": 0.9840, "I-AUROC": 0.9986, "P-AUPRO": 0.9886, "Threshold": 20.8826},
    "terminalblock": {"S-AUROC": 0.9701, "I-AUROC": 0.9985, "P-AUPRO": 0.9576, "Threshold": 19.0404},
    "toothbrush": {"S-AUROC": 0.8894, "I-AUROC": 0.9929, "P-AUPRO": 0.9433, "Threshold": 25.2239},
    "toy": {"S-AUROC": 0.8652, "I-AUROC": 0.9319, "P-AUPRO": 0.8768, "Threshold": 21.0941},
    "toy_brick": {"S-AUROC": 0.8664, "I-AUROC": 0.9895, "P-AUPRO": 0.8850, "Threshold": 22.0600},
    "transistor1": {"S-AUROC": 0.8999, "I-AUROC": 0.9903, "P-AUPRO": 0.8194, "Threshold": 22.5007},
    "u_block": {"S-AUROC": 0.9209, "I-AUROC": 0.9950, "P-AUPRO": 0.9584, "Threshold": 17.5335},
    "usb": {"S-AUROC": 0.9558, "I-AUROC": 0.9980, "P-AUPRO": 0.9803, "Threshold": 19.6187},
    "usb_adaptor": {"S-AUROC": 0.8617, "I-AUROC": 0.9952, "P-AUPRO": 0.8479, "Threshold": 19.1885},
    "vcpill": {"S-AUROC": 0.8794, "I-AUROC": 0.9864, "P-AUPRO": 0.7087, "Threshold": 19.5500},
    "wooden_beads": {"S-AUROC": 0.8725, "I-AUROC": 0.9854, "P-AUPRO": 0.8479, "Threshold": 20.9424},
    "woodstick": {"S-AUROC": 0.9194, "I-AUROC": 0.9803, "P-AUPRO": 0.8299, "Threshold": 21.0806},
    "zipper": {"S-AUROC": 0.9770, "I-AUROC": 0.9909, "P-AUPRO": 0.8876, "Threshold": 17.7488},
}

CATEGORIES = sorted(list(BENCHMARK_DATA.keys()))

# ==============================================================================
# 3. Model Architecture
# ==============================================================================
class WideResNet50FeatureExtractor(nn.Module):
    """
    Extracts concatenated patch representations from Layer 2 and Layer 3
    of Wide ResNet-50-2 for 512x512 inputs.
    Output: [B, 1536, 64, 64] -> 4,096 patch feature vectors per image.
    """
    def __init__(self, resnet):
        super().__init__()
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.stem(x)
        f2 = self.layer2(x)   # [B, 512, 64, 64]
        f3 = self.layer3(f2)  # [B, 1024, 32, 32]

        f2 = self.avg_pool(f2)
        f3 = self.avg_pool(f3)

        # Bilinear interpolation of Layer 3 to match Layer 2 spatial resolution (64x64)
        f3 = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return torch.cat([f2, f3], dim=1)  # [B, 1536, 64, 64]


# ==============================================================================
# 4. Model Loading with Caching
# ==============================================================================
MODEL_REPO = "JayedAnsari/realiad-patchcore"
MODEL_FILENAME = "realiad_patchcore_model.pt"


@st.cache_resource(show_spinner="Loading Real-IAD PatchCore model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
    except Exception as e:
        return (
            None, None, None, None, None, device,
            f"Failed to download model artifact: {e}"
        )

    artifact = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    base_model = wide_resnet50_2(weights=None)
    feature_extractor = WideResNet50FeatureExtractor(base_model)

    feature_extractor.load_state_dict(
        artifact["backbone_state_dict"]
    )

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    memory_banks = artifact["memory_banks"]
    thresholds = artifact["thresholds"]

    config = {
        "dataset": artifact.get("dataset", "Real-IAD"),
        "backbone": artifact.get("backbone", "Wide ResNet-50-2"),
        "feature_layers": artifact.get(
            "feature_layers",
            ["layer2", "layer3"]
        ),
        "feature_dim": artifact.get("feature_dim", 1536),
        "input_size": artifact.get("input_size", 512),
        "sampling_ratio": artifact.get("sampling_ratio", 0.01),
        "distance_metric": artifact.get(
            "distance_metric",
            "euclidean"
        ),
        "normalize_mean": artifact.get(
            "normalize_mean",
            [0.485, 0.456, 0.406]
        ),
        "normalize_std": artifact.get(
            "normalize_std",
            [0.229, 0.224, 0.225]
        ),
        "gaussian_kernel": artifact.get(
            "gaussian_kernel",
            33
        ),
        "gaussian_sigma": artifact.get(
            "gaussian_sigma",
            4.0
        ),
    }

    transform = T.Compose([
        T.Resize(
            (config["input_size"], config["input_size"])
        ),
        T.ToTensor(),
        T.Normalize(
            mean=config["normalize_mean"],
            std=config["normalize_std"]
        )
    ])

    return (
        feature_extractor,
        memory_banks,
        thresholds,
        transform,
        config,
        device,
        None
    )


# ==============================================================================
# 5. Inference Pipeline
# ==============================================================================
def predict(image: Image.Image, category: str, feature_extractor, memory_banks, thresholds, transform, config, device):
    input_size = config["input_size"]
    g_kernel = (config["gaussian_kernel"], config["gaussian_kernel"])
    g_sigma = (config["gaussian_sigma"], config["gaussian_sigma"])
    
    img_rgb = image.convert("RGB").resize((input_size, input_size))
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    if category not in memory_banks or category not in thresholds:
        raise KeyError(f"Category '{category}' not found in model memory banks.")
        
    mem_bank = memory_banks[category].to(device=device, dtype=torch.float32)
    threshold = float(thresholds[category])
    
    with torch.no_grad():
        feats = feature_extractor(input_tensor).float() # [1, 1536, 64, 64]
        B, C, H, W = feats.shape
        patches = feats.permute(0, 2, 3, 1).reshape(-1, C)  # [4096, 1536]
        
        # Chunked Euclidean distance computation
        min_dists = []
        chunk_size = 256
        for i in range(0, patches.shape[0], chunk_size):
            chunk = patches[i : i + chunk_size]
            dists = torch.cdist(chunk, mem_bank)
            min_dists.append(dists.min(dim=1).values)
            
        patch_min_dists = torch.cat(min_dists)  # [4096]
        score = float(patch_min_dists.max().cpu().item())
        
        # 64x64 Anomaly map
        amap_64 = patch_min_dists.reshape(1, 1, H, W)
        
        # Bilinear interpolation up to input_size
        amap_up = F.interpolate(amap_64, size=(input_size, input_size), mode="bilinear", align_corners=False)
        
        # Gaussian blur for visualization heatmap
        smoothed_map = TF.gaussian_blur(amap_up, kernel_size=g_kernel, sigma=g_sigma).squeeze().cpu().numpy()
        
    is_anomaly = score >= threshold
    
    # Generate 0.70 * orig + 0.30 * inferno heatmap overlay
    orig_np = np.array(img_rgb) / 255.0
    v_min, v_max = float(smoothed_map.min()), float(smoothed_map.max())
    norm_map = np.clip((smoothed_map - v_min) / (v_max - v_min + 1e-8), 0.0, 1.0)
    heatmap_rgb = cm.inferno(norm_map)[..., :3]
    overlay_np = np.clip(0.70 * orig_np + 0.30 * heatmap_rgb, 0.0, 1.0)
    
    return {
        "category": category,
        "score": score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "anomaly_map": smoothed_map,
        "orig_img": orig_np,
        "overlay_img": overlay_np,
    }


# ==============================================================================
# 6. Streamlit User Interface
# ==============================================================================
def main():
    # Load Model
    feature_extractor, memory_banks, thresholds, transform, config, device, err = load_model()

    if err:
        st.error(err)
        return

    model_categories = sorted(memory_banks.keys())

    # --------------------------------------------------------------------------
    # SIDEBAR: Model Information & Benchmark Metrics
    # --------------------------------------------------------------------------
    with st.sidebar:
        sampling_val = config.get("sampling_ratio", 0.01)
        sampling_pct = f"{sampling_val * 100:.1f}%" if isinstance(sampling_val, (int, float)) else str(sampling_val)
        dist_metric = str(config.get("distance_metric", "euclidean")).capitalize()
        feat_layers = config.get("feature_layers", ["layer2", "layer3"])
        feat_layers_str = " + ".join([l.capitalize() for l in feat_layers]) if isinstance(feat_layers, list) else str(feat_layers)

        st.markdown("### MODEL INFORMATION")
        st.markdown(f"""
        - **Dataset:** {config.get('dataset', 'Real-IAD')}
        - **Backbone:** {config.get('backbone', 'Wide ResNet-50-2')}
        - **Feature Layers:** {feat_layers_str}
        - **Feature Dimension:** {config.get('feature_dim', 1536)}
        - **Input Resolution:** {config['input_size']} × {config['input_size']}
        - **Patch Grid:** 64 × 64 (4,096 patches)
        - **Memory Bank Sampling:** {sampling_pct}
        - **Distance Metric:** {dist_metric}
        """)

        st.markdown("---")
        st.markdown("### BENCHMARK PERFORMANCE")
        
        st.markdown("#### Overall Mean (Test Set)")
        st.markdown("""
        - **S-AUROC:** `0.9056`
        - **I-AUROC:** `0.9906`
        - **P-AUPRO:** `0.8803`
        """)

        st.caption("Metrics reported on the Real-IAD test set; they are not calculated from the uploaded image.")

    # --------------------------------------------------------------------------
    # MAIN UI: Inspection, Prediction & Visualizations
    # --------------------------------------------------------------------------
    st.markdown('<div class="main-title">Real-IAD Industrial Defect Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">PATCHCORE-STYLE ANOMALY LOCALIZATION & INSPECTION</div>', unsafe_allow_html=True)
    st.write("Detect and localize industrial defects using a Wide ResNet-50-2 feature extractor and category-specific PatchCore memory banks.")
    
    st.markdown("---")

    # Inputs on Main UI
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_category = st.selectbox(
            "Category",
            options=model_categories,
            index=model_categories.index("zipper") if "zipper" in model_categories else 0
        )
    with col2:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    # Show selected category benchmark stats in sidebar dynamically
    with st.sidebar:
        st.markdown("### SELECTED CATEGORY")
        st.caption(selected_category.upper())
        cat_b = BENCHMARK_DATA.get(selected_category, {})
        s_auroc = f"{cat_b['S-AUROC']:.4f}" if "S-AUROC" in cat_b else "N/A"
        i_auroc = f"{cat_b['I-AUROC']:.4f}" if "I-AUROC" in cat_b else "N/A"
        p_aupro = f"{cat_b['P-AUPRO']:.4f}" if "P-AUPRO" in cat_b else "N/A"
        model_thresh = f"{thresholds[selected_category]:.4f}" if selected_category in thresholds else "N/A"
        
        st.markdown(f"""
        - **S-AUROC:** `{s_auroc}`
        - **I-AUROC:** `{i_auroc}`
        - **P-AUPRO:** `{p_aupro}`
        - **Threshold:** `{model_thresh}`
        """)

    # Prediction & Visualizations on Main UI
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        if analyze_btn or "res" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name or st.session_state.get("cat") != selected_category:
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["cat"] = selected_category
            
            with st.spinner("Running inference..."):
                res = predict(
                    image=image,
                    category=selected_category,
                    feature_extractor=feature_extractor,
                    memory_banks=memory_banks,
                    thresholds=thresholds,
                    transform=transform,
                    config=config,
                    device=device
                )
            st.session_state["res"] = res

        if "res" in st.session_state:
            res = st.session_state["res"]
            
            st.markdown("---")
            st.subheader("Prediction")
            
            if res["is_anomaly"]:
                st.markdown('<div class="status-defective">STATUS: DEFECTIVE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-normal">STATUS: NORMAL</div>', unsafe_allow_html=True)

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Anomaly Score", f"{res['score']:.2f}")
            col_m2.metric("Threshold", f"{res['threshold']:.2f}")

            st.markdown("---")

            # Image Visualizations
            st.subheader("Visualizations")
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown("**Original Image**")
                st.image(res["orig_img"], use_container_width=True)
                
            with col_img2:
                st.markdown("**Anomaly Heatmap**")
                fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0E1113')
                ax.set_facecolor('#0E1113')
                im = ax.imshow(res["anomaly_map"], cmap="inferno")
                ax.axis("off")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.yaxis.set_tick_params(color='#92999F')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#92999F')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("**Localization Overlay**")
            col_left, col_overlay, col_right = st.columns([1, 2, 1])
            with col_overlay:
                st.image(res["overlay_img"], use_container_width=True)

            # Detection Details
            st.markdown("---")
            st.subheader("Detection Details")
            st.markdown(f"""
            - **Category:** `{res['category']}`
            - **Anomaly Score:** `{res['score']:.2f}`
            - **Threshold:** `{res['threshold']:.2f}`
            - **Status:** `{'DEFECTIVE' if res['is_anomaly'] else 'NORMAL'}`
            """)


if __name__ == "__main__":
    main()
