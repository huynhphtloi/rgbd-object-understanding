"""
Streamlit web demo — capture RGB + estimate depth + run full pipeline.
"""
import os
import sys
import json
import tempfile
import io

import cv2
import numpy as np
import streamlit as st
import yaml

# Add repo to path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from src.pipeline import Pipeline
from src.depth.depth_utils import CameraIntrinsics


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def estimate_depth_midas(rgb_img):
    """Estimate depth from RGB using MiDaS (lightweight)."""
    try:
        import torch
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize
        from midas.model_loader import load_model
    except ImportError:
        st.error("MiDaS not installed. Install: pip install timm opencv-python torch torchvision")
        return None

    # Load MiDaS small (fastest)
    model_type = "MiDaS_small"
    device = "cpu"

    model, transforms = load_model(model_type, "weights", device, False)
    model.eval()

    # Prepare input
    img_h, img_w = rgb_img.shape[:2]
    input_batch = transforms(rgb_img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(img_h, img_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    # Normalize to 16-bit depth (0-65535)
    depth_16 = ((depth - depth.min()) / (depth.max() - depth.min()) * 65535).astype(np.uint16)
    return depth_16


def run_pipeline(rgb, depth, intrinsics, cfg):
    """Run full RGB-D pipeline."""
    # Parse intrinsics
    intr = CameraIntrinsics(**intrinsics)

    # Initialize pipeline
    pipeline = Pipeline(cfg)

    # Run
    result = pipeline.run(
        rgb=rgb,
        depth=depth,
        intr=intr,
        synthetic=False,
    )

    return result


st.set_page_config(page_title="RGB-D Scene Understanding", layout="wide")
st.title("🎥 RGB-D Scene Understanding Demo")

cfg = load_config(os.path.join(REPO, "config", "demo.yaml"))

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Capture", "📊 Results", "⚙️ Settings"])

with tab1:
    st.subheader("Input RGB-D")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Option 1: Camera Capture**")
        picture = st.camera_input("Take a photo")

        if picture:
            rgb = cv2.imdecode(np.frombuffer(picture.read(), np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption="RGB captured", use_column_width=True)
            st.session_state.rgb = rgb

    with col2:
        st.write("**Option 2: Upload Files**")
        rgb_file = st.file_uploader("Upload RGB PNG", type=["png", "jpg"])
        depth_file = st.file_uploader("Upload Depth PNG (16-bit)", type=["png"])

        if rgb_file and depth_file:
            rgb = cv2.imdecode(np.frombuffer(rgb_file.read(), np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imdecode(np.frombuffer(depth_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            st.image(rgb, caption="RGB uploaded", use_column_width=True)
            st.session_state.rgb = rgb
            st.session_state.depth = depth

    # Intrinsics
    st.write("**Camera Intrinsics**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fx = st.number_input("fx", value=585.0)
    with col2:
        fy = st.number_input("fy", value=585.0)
    with col3:
        cx = st.number_input("cx", value=320.0)
    with col4:
        cy = st.number_input("cy", value=240.0)

    depth_scale = st.number_input("Depth scale (mm→m)", value=0.001)

    st.session_state.intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "depth_scale": depth_scale}

with tab2:
    if st.button("🚀 Run Pipeline", key="run"):
        if "rgb" not in st.session_state:
            st.error("❌ No RGB image!")
        elif "depth" not in st.session_state:
            st.warning("⏳ Estimating depth from RGB...")
            with st.spinner("Running MiDaS..."):
                depth = estimate_depth_midas(st.session_state.rgb)
                if depth is None:
                    st.stop()
                st.session_state.depth = depth
                st.success("✅ Depth estimated!")

        with st.spinner("Running pipeline..."):
            try:
                result = run_pipeline(
                    st.session_state.rgb,
                    st.session_state.depth,
                    st.session_state.intrinsics,
                    cfg,
                )

                st.session_state.result = result
                st.success("✅ Pipeline complete!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if "result" in st.session_state:
        result = st.session_state.result

        st.subheader("📈 Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Objects Detected", len(result.instances))
        with col2:
            st.metric("Relations Found", len(result.relations))
        with col3:
            if result.counts:
                st.metric("Count Status", "✅ Valid")
        with col4:
            st.metric("Scene Complexity", f"{len(result.scene.get('nodes', []))} nodes")

        st.divider()

        # Tabs for results
        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(
            ["Masks", "Dimensions", "Relations", "JSON"]
        )

        with res_tab1:
            st.write("**Instance Segmentation**")
            # TODO: visualize masks
            st.json({
                "objects": len(result.instances),
                "masks": "Visualize here"
            })

        with res_tab2:
            st.write("**3D Dimensions (cm)**")
            dims_data = []
            for i, inst in enumerate(result.instances):
                if inst.measurement:
                    dims_data.append({
                        "ID": i,
                        "Width (cm)": round(inst.measurement.width_cm, 2),
                        "Height (cm)": round(inst.measurement.height_cm, 2),
                        "Depth (cm)": round(inst.measurement.depth_cm, 2),
                    })
            if dims_data:
                st.dataframe(dims_data, use_container_width=True)
            else:
                st.info("No dimensions estimated")

        with res_tab3:
            st.write("**Spatial Relations**")
            rel_data = []
            for rel in result.relations:
                rel_data.append({
                    "Object A": rel.get("obj_a_id"),
                    "Relation": rel.get("relation_type"),
                    "Object B": rel.get("obj_b_id"),
                })
            if rel_data:
                st.dataframe(rel_data, use_container_width=True)
            else:
                st.info("No relations found")

        with res_tab4:
            st.write("**Scene Graph (JSON)**")
            st.json(result.scene)

with tab3:
    st.write("**Pipeline Config**")
    st.json(cfg)

    if st.button("Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    st.sidebar.info(
        """
        **RGB-D Scene Understanding Web Demo**

        1. Capture or upload RGB image
        2. Optionally upload depth (auto-estimate if missing)
        3. Set camera intrinsics
        4. Click "Run Pipeline"
        5. View results (masks, dimensions, relations, scene graph)
        """
    )
