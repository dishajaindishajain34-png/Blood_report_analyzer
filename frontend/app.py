"""
app.py — Phase 2 Streamlit UI for AI Blood Report Analyzer.

Tabs:
  1. Analyze    — single image or batch upload + full pipeline
  2. History    — SQLite patient records browser
  3. About      — project info + model status
"""

from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
from src.utils import project_path
import io

# ── Page config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Blood Report Analyzer",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }
.main-header {
    background: linear-gradient(135deg, #0a6cb4 0%, #10b981 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem; text-align: center;
}
.main-header h1 { color: #fff; font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0; font-size: 1rem; }
.metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 1.2rem 1.4rem; text-align: center;
}
.metric-card .val  { font-size: 2rem; font-weight: 700; }
.metric-card .lbl  { font-size: 0.8rem; color: #8b949e; margin-top: 0.2rem; }
.badge-normal   { background:#0d3a25; color:#3fb950; padding:4px 14px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.badge-abnormal { background:#3d1a1a; color:#f85149; padding:4px 14px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.flag-row { background:#1f1a0e; border-left:3px solid #d29922; padding:0.6rem 1rem; border-radius:6px; margin:0.3rem 0; font-size:0.85rem; }
.ok-row   { background:#0d2217; border-left:3px solid #3fb950; padding:0.6rem 1rem; border-radius:6px; margin:0.3rem 0; font-size:0.85rem; }
.section-title { color:#58a6ff; font-size:1rem; font-weight:600; margin:1.2rem 0 0.5rem; border-bottom:1px solid #30363d; padding-bottom:0.4rem; }
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg,#0a6cb4,#0d9488); color:#fff;
    border:none; border-radius:8px; font-weight:600; padding:0.6rem 1.5rem;
}
div[data-testid="stButton"] > button:hover { opacity:0.88; }
</style>
""", unsafe_allow_html=True)


# ── Pre-warm models on startup ──────────────────────────────────────────────────────
@st.cache_resource
def prewarm_models():
    try:
        from src.detect import warmup_model
        warmup_model()
    except Exception as e:
        print(f"Could not pre-warm YOLO model: {e}")
    try:
        from src.anemia_image_model import _load_image_model
        _load_image_model()
    except Exception as e:
        pass
    try:
        from src.anemia_model import load_anemia_pipeline
        load_anemia_pipeline()
    except Exception as e:
        pass

prewarm_models()


# ── Hero ───────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🩸 AI Anemia & Blood Report Analyzer</h1>
  <p>Anemia Severity Profiling · Hemoglobin & Hematocrit Estimation · YOLOv8 Cell Detection · Patient History</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_thresh = st.slider("Detection Confidence", 0.20, 0.80, 0.35, 0.05)
    iou_thresh  = st.slider("NMS IOU Threshold",    0.20, 0.70, 0.40, 0.05)
    st.divider()

    st.markdown("### 👤 Patient Info (optional)")
    patient_name = st.text_input("Patient Name")
    patient_id   = st.text_input("Patient ID")
    patient_age  = st.number_input("Age", 0, 120, 0)

    st.divider()
    st.markdown("### 🩸 Laboratory CBC Inputs (ML Anemia Model)")
    enable_ml_anemia = st.checkbox("Enable Secondary ML Anemia Stream", value=True)
    with st.expander("Enter Patient Lab CBC Parameters", expanded=False):
        cbc_gender_str = st.radio("Gender", ["Female", "Male"], index=0, horizontal=True)
        cbc_hb   = st.number_input("Hemoglobin (g/dL)", min_value=1.0, max_value=25.0, value=12.5, step=0.1)
        cbc_mcv  = st.number_input("MCV (fL)", min_value=30.0, max_value=150.0, value=85.0, step=0.5)
        cbc_mch  = st.number_input("MCH (pg)", min_value=10.0, max_value=50.0, value=28.0, step=0.5)
        cbc_mchc = st.number_input("MCHC (g/dL)", min_value=15.0, max_value=45.0, value=33.0, step=0.5)

    st.divider()
    st.markdown("### 📋 Pipeline Status")
    wbc_model_path = project_path("models", "resnet50_wbc", "wbc_classifier_sklearn.pth")
    anemia_model_path = project_path("models", "anemia", "anemia_model.joblib")
    if wbc_model_path.exists():
        st.success("WBC Classifier: sklearn ✓")
    else:
        st.warning("WBC Classifier: Placeholder mode")
    st.success("YOLOv8: best.pt ✓")
    if anemia_model_path.exists():
        st.success("ML Anemia: Random Forest ✓")
    else:
        st.warning("ML Anemia: Model Missing")

# ── Tabs ───────────────────────────────────────────────────────────────────────────
tab_analyze, tab_history, tab_about = st.tabs(["🔬 Analyze", "📋 History", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ══════════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    col_up, col_info = st.columns([2, 1])

    with col_up:
        upload_mode = st.radio("Mode", ["Single Image", "Batch (multiple images)"],
                               horizontal=True, label_visibility="collapsed")
        if upload_mode == "Single Image":
            uploaded = st.file_uploader("Upload blood smear image",
                                        type=["jpg","jpeg","png"], label_visibility="collapsed")
            files = [uploaded] if uploaded else []
        else:
            uploaded_multi = st.file_uploader("Upload multiple blood smear images",
                                              type=["jpg","jpeg","png"], accept_multiple_files=True,
                                              label_visibility="collapsed")
            files = uploaded_multi or []

    with col_info:
        st.markdown("""
        **Supported formats:** JPEG · PNG  
        **Batch mode:** upload 2-10 images for averaged results  
        **Output:** Cell counts · WBC differential · PDF report
        """)
        use_sample = st.button("🔬 Use Sample Image")

    # Load images into session_state for persistence across rerenders
    if "loaded_images" not in st.session_state:
        st.session_state.loaded_images = []

    if use_sample:
        sp = project_path("data", "sample_images", "sample.jpg")
        if sp.exists():
            st.session_state.loaded_images = [Image.open(str(sp)).convert("RGB")]
            st.info("Using sample: BloodImage_00001.jpg (BCCD dataset)")
    elif files:
        st.session_state.loaded_images = [Image.open(f).convert("RGB") for f in files]

    images = st.session_state.loaded_images

    # Show thumbnail preview
    if images:
        pcols = st.columns(min(len(images), 5))
        for i, (col, img) in enumerate(zip(pcols, images)):
            with col:
                st.image(img, caption=f"Image {i+1}", use_container_width=True)


    if images and st.button("▶ Run Analysis"):
        from src.batch_analyze import run_batch
        from src.anemia_model import predict_anemia_ml

        n = len(images)
        is_batch = n > 1
        prog = st.progress(0.0)
        status = st.empty()

        def _progress_cb(step: int, total: int, msg: str):
            try:
                frac = min(1.0, (step + 1) / total)
            except Exception:
                frac = 0.0
            prog.progress(frac)
            status.text(msg)

        # Run secondary ML Anemia prediction on user-entered laboratory CBC parameters if enabled
        ml_anemia_res = None
        cbc_input_payload = {}
        if enable_ml_anemia:
            cbc_input_payload = {
                "Gender": 0 if cbc_gender_str == "Female" else 1,
                "Hemoglobin": cbc_hb,
                "MCV": cbc_mcv,
                "MCH": cbc_mch,
                "MCHC": cbc_mchc,
            }
            ml_anemia_res = predict_anemia_ml(cbc_input_payload)

        # Run Image-based Anemia Screening
        from src.anemia_image_model import predict_image_anemia
        from src.anemia_fusion import predict_fusion
        img_anemia_res = predict_image_anemia(images[0])

        # Run Fusion prediction
        image_prob = img_anemia_res.get("probability") if img_anemia_res.get("status") == "success" else None
        cbc_prob = ml_anemia_res.get("probability") if (ml_anemia_res and ml_anemia_res.get("status") == "success") else None
        fusion_res = predict_fusion(image_prob, cbc_prob)

        # Run the centralized batch pipeline
        result = run_batch(
            images,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            progress_callback=_progress_cb,
            anemia_ml_result=ml_anemia_res,
            anemia_image_result=img_anemia_res,
            anemia_fusion_result=fusion_res
        )

        # Unpack result
        aid = result.analysis_id
        n = result.n_images
        is_batch = n > 1
        avg_raw = result.avg_raw
        avg_diff = result.avg_differential
        avg_conf = result.avg_confidence
        cal = result.avg_calibrated
        flags = result.flags
        first_orig = result.original_pil
        first_ann = result.annotated_pil
        last_is_estimated = result.is_estimated_diff
        pdf_path = result.pdf_path

        # Save to DB
        try:
            from database.connection import get_session_factory, init_db
            from database.crud import create_record
            init_db()
            SessionLocal = get_session_factory()
            with SessionLocal() as db:
                create_record(db, {
                    "analysis_id": aid,
                    "patient_name": patient_name or None,
                    "patient_id":   patient_id or None,
                    "patient_age":  int(patient_age) if patient_age else None,
                    "raw_rbc": avg_raw.get("rbc", 0), "raw_wbc": avg_raw.get("wbc", 0),
                    "raw_platelet": avg_raw.get("platelet", 0),
                    "total_detected": sum(avg_raw.values()),
                    "avg_confidence": avg_conf,
                    "cal_rbc": cal.get("rbc") if isinstance(cal, dict) else None,
                    "cal_wbc": cal.get("wbc") if isinstance(cal, dict) else None,
                    "cal_platelet": cal.get("platelet") if isinstance(cal, dict) else None,
                    "wbc_differential": avg_diff,
                    "is_estimated_diff": last_is_estimated,
                    "overall_result": flags.overall if flags else "Unknown",
                    "flag_summary": flags.summary if flags else [],
                    "anemia_ml_prediction": ml_anemia_res.get("prediction") if ml_anemia_res else None,
                    "anemia_ml_probability": ml_anemia_res.get("probability") if ml_anemia_res else None,
                    "anemia_ml_model_version": ml_anemia_res.get("model_version") if ml_anemia_res else None,
                    "cbc_inputs_json": cbc_input_payload if ml_anemia_res else None,
                    "image_anemia_prediction": img_anemia_res.get("prediction") if img_anemia_res else None,
                    "image_anemia_probability": img_anemia_res.get("probability") if img_anemia_res else None,
                    "cbc_anemia_prediction": ml_anemia_res.get("prediction") if ml_anemia_res else None,
                    "cbc_anemia_probability": ml_anemia_res.get("probability") if ml_anemia_res else None,
                    "fusion_anemia_prediction": fusion_res.get("fusion_prediction") if fusion_res else None,
                    "fusion_anemia_probability": fusion_res.get("fusion_probability") if fusion_res else None,
                    "fusion_model_version": fusion_res.get("model_version") if fusion_res else None,
                    "pdf_path": str(pdf_path) if pdf_path else None,
                    "image_filename": "sample.jpg" if use_sample else (files[0].name if files else ""),
                })
        except Exception as e:
            st.warning(f"DB save skipped: {e}")

        prog.progress(1.0)
        status.empty()
        prog.empty()

        # ── Results ────────────────────────────────────────────────────────────
        st.divider()
        r1, r2, r3 = st.columns([2, 1, 1])
        with r1:
            badge = "badge-normal" if flags and flags.overall == "Normal" else "badge-abnormal"
            st.markdown(f"**Analysis ID:** `{aid}`")
            st.markdown(f"<span class='{badge}'>{flags.overall if flags else 'Unknown'}</span>", unsafe_allow_html=True)
            if is_batch:
                st.caption(f"Averaged across {n} images")
        with r2:
            if pdf_path and Path(str(pdf_path)).exists():
                st.download_button("📄 Download PDF", Path(str(pdf_path)).read_bytes(),
                                   f"{aid}.pdf", "application/pdf")
        with r3:
            if st.button("🔄 New Analysis"):
                st.session_state.loaded_images = []
                st.rerun()

        # Images
        st.markdown('<div class="section-title">Blood Smear Analysis</div>', unsafe_allow_html=True)
        ic1, ic2 = st.columns(2)
        with ic1:
            st.image(first_orig, caption="Original", use_container_width=True)
        with ic2:
            st.image(first_ann,  caption="YOLOv8 Detections", use_container_width=True)

        # ── A. BLOOD CELL CLASSIFICATION & COUNTS ─────────────────────────────────
        st.markdown('<div class="section-title">🩸 Blood Cell Classification Counts (smear raw counts)</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        cells = [
            ("RBC (Red Blood Cells)",   avg_raw.get("rbc", 0),      "#ef4444"),
            ("WBC (White Blood Cells)",   avg_raw.get("wbc", 0),      "#3b82f6"),
            ("Platelets",                 avg_raw.get("platelet", 0), "#f59e0b"),
        ]
        for col, (name, val, color) in zip([mc1, mc2, mc3], cells):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val" style="color:{color}">{val:,}</div>
                  <div class="lbl">{name} detected</div>
                </div>""", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""
            <div class="metric-card">
              <div class="val" style="color:#8b949e">{avg_conf:.0%}</div>
              <div class="lbl">YOLOv8 Avg Confidence</div>
            </div>""", unsafe_allow_html=True)

        # ── B. IMAGE-BASED ANEMIA ANALYSIS ────────────────────────────────────────
        st.markdown('<div class="section-title">🖼️ Image-Based Anemia Screening (AneRBC model)</div>', unsafe_allow_html=True)
        im1, im2, im3, im4 = st.columns(4)
        if img_anemia_res:
            img_pred = img_anemia_res.get("prediction", "Undetermined")
            img_prob = img_anemia_res.get("probability", 0.0)
            img_ver = img_anemia_res.get("model_version", "1.0.0")
            img_status = img_anemia_res.get("status", "success")

            if img_status == "success":
                img_color = "#f85149" if img_pred == "Anemic" else "#3fb950"
                with im1:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="val" style="color:{img_color}; font-size:1.4rem;">{img_pred}</div>
                      <div class="lbl">Image Prediction</div>
                    </div>""", unsafe_allow_html=True)
                with im2:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="val" style="color:#58a6ff">{img_prob:.1%}</div>
                      <div class="lbl">Anemia Probability</div>
                    </div>""", unsafe_allow_html=True)
                with im3:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="val" style="color:#d29922; font-size:1.1rem;">MobileNetV3</div>
                      <div class="lbl">Model Architecture</div>
                    </div>""", unsafe_allow_html=True)
                with im4:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="val" style="color:#8b949e">v{img_ver}</div>
                      <div class="lbl">Model Version</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning(img_anemia_res.get("message", "Model not available"))
        else:
            st.info("Image-based model not evaluated.")

        # ── C. CBC-BASED ANEMIA ANALYSIS ──────────────────────────────────────────
        if ml_anemia_res and ml_anemia_res.get("status") == "success":
            st.markdown('<div class="section-title">🤖 CBC-Based Anemia Screening Model (laboratory CBC stream)</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            is_anemic_pred = ml_anemia_res.get("is_anemic", False)
            ml_pred_str    = ml_anemia_res.get("prediction", "Normal")
            ml_prob_val    = ml_anemia_res.get("probability", 0.0)
            ml_model_name  = ml_anemia_res.get("model_name", "Random Forest")
            ml_ver         = ml_anemia_res.get("model_version", "1.0.0")

            pred_color = "#f85149" if is_anemic_pred else "#3fb950"
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val" style="color:{pred_color}; font-size:1.4rem;">{ml_pred_str}</div>
                  <div class="lbl">CBC ML Prediction</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val" style="color:#58a6ff">{ml_prob_val:.1%}</div>
                  <div class="lbl">Anemia Probability</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val" style="color:#d29922; font-size:1.1rem;">{ml_model_name}</div>
                  <div class="lbl">Model Architecture</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val" style="color:#8b949e">v{ml_ver}</div>
                  <div class="lbl">Model Version</div>
                </div>""", unsafe_allow_html=True)

            st.caption(f"**Laboratory Inputs:** Gender: {cbc_gender_str} | Hb: {cbc_hb} g/dL | MCV: {cbc_mcv} fL | MCH: {cbc_mch} pg | MCHC: {cbc_mchc} g/dL")

        # ── D. FINAL FUSED ANEMIA ASSESSMENT ──────────────────────────────────────
        st.markdown('<div class="section-title">🏆 FINAL MULTIMODAL ANEMIA ASSESSMENT</div>', unsafe_allow_html=True)
        if fusion_res:
            fuse_pred = fusion_res.get("fusion_prediction", "Undetermined")
            fuse_prob = fusion_res.get("fusion_probability", 0.0)
            fuse_strategy = fusion_res.get("fusion_strategy", "none")
            fuse_ver = fusion_res.get("model_version", "1.0.0")

            fuse_color = "#f85149" if fuse_pred == "Anemic" else "#3fb950"
            strategy_desc = {
                "learned_logistic_regression": "Learned Multimodal Fusion (Logistic Regression Model)",
                "learned_lr": "Learned Multimodal Fusion (Logistic Regression Model)",
                "average_fallback": "Fallback Fusion — trained fusion model unavailable (Average of Image & CBC Probabilities)",
                "image_only": "Image Modality Only",
                "cbc_only": "CBC Modality Only",
                "none": "No inputs available"
            }.get(fuse_strategy, fuse_strategy)

            f1, f2, f3 = st.columns([2, 2, 1])
            with f1:
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid {fuse_color}; background-color: #1a1f29;">
                  <div class="val" style="color:{fuse_color}; font-size:2rem; font-weight: 800;">{fuse_pred.upper()}</div>
                  <div class="lbl">FINAL MULTIMODAL ASSESSMENT</div>
                </div>""", unsafe_allow_html=True)
            with f2:
                st.markdown(f"""
                <div class="metric-card" style="border: 1px solid #30363d;">
                  <div class="val" style="color:#58a6ff; font-size:1.8rem;">{fuse_prob:.1%}</div>
                  <div class="lbl">Fused Anemia Probability</div>
                </div>""", unsafe_allow_html=True)
            with f3:
                st.markdown(f"""
                <div class="metric-card" style="border: 1px solid #30363d;">
                  <div class="val" style="color:#d29922; font-size:1.1rem; padding: 0.5rem 0;">v{fuse_ver}</div>
                  <div class="lbl">Fusion Version</div>
                </div>""", unsafe_allow_html=True)

            st.caption(f"**Fusion Method:** {strategy_desc} (Combines blood smear image features and CBC lab values)")
        else:
            st.info("Fusion assessment not generated.")

        # WBC Differential donut
        st.markdown('<div class="section-title">WBC Differential</div>', unsafe_allow_html=True)
        dc1, dc2 = st.columns([1, 1])
        with dc1:
            labels = list(avg_diff.keys())
            vals   = list(avg_diff.values())
            colors = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6"]
            fig = go.Figure(go.Pie(
                labels=labels, values=vals, hole=0.55,
                marker_colors=colors,
                textinfo="label+percent", textfont_size=12,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3", showlegend=False, height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                annotations=[dict(text="Sklearn" if not last_is_estimated else "Est.",
                                  x=0.5, y=0.5, font_size=13, showarrow=False,
                                  font_color="#8b949e")]
            )
            st.plotly_chart(fig, use_container_width=True)
        with dc2:
            df_diff = pd.DataFrame(
                [(cls, f"{pct:.0f}%") for cls, pct in avg_diff.items()],
                columns=["Subtype", "Percentage"]
            )
            st.dataframe(df_diff, hide_index=True, use_container_width=True)
            if last_is_estimated:
                st.caption("⚠️ Estimated from healthy reference ratios — not from real classification.")
            else:
                st.warning(
                    "⚠️ **Accuracy Notice:** WBC subtype percentages are **indicative only**. "
                    "The classifier was trained on ~415 pseudo-labeled crops. "
                    "Values may not reflect true clinical differential. "
                    "A pathologist-validated dataset is required for clinical accuracy."
                )

        # Clinical flags
        st.markdown('<div class="section-title">Clinical Findings</div>', unsafe_allow_html=True)
        for line in flags.summary:
            if any(w in line for w in ["High","Low","Possible"]):
                st.markdown(f'<div class="flag-row">⚠ {line}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ok-row">✓ {line}</div>', unsafe_allow_html=True)

        st.caption("DISCLAIMER: AI-generated analysis. Not a medical diagnosis. Always consult a certified pathologist.")


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="section-title">Patient Analysis History</div>', unsafe_allow_html=True)

    try:
        from database.connection import get_session_factory, init_db
        from database.crud import get_all_records, get_stats
        init_db()
        SessionLocal = get_session_factory()

        with SessionLocal() as db:
            stats   = get_stats(db)
            records = get_all_records(db, limit=200)

        # Stats row
        s1, s2, s3, s4 = st.columns(4)
        for col, (label, val, color) in zip(
            [s1, s2, s3, s4],
            [("Total Analyses", stats["total_analyses"], "#58a6ff"),
             ("Normal",         stats["normal"],         "#3fb950"),
             ("Abnormal",       stats["abnormal"],       "#f85149"),
             ("Abnormal %",     f"{stats['abnormal_pct']}%", "#d29922")]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val" style="color:{color}">{val}</div>
                  <div class="lbl">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()

        if not records:
            st.info("No analyses recorded yet. Run an analysis in the Analyze tab first.")
        else:
            # Search/filter
            search = st.text_input("Search by patient name or ID", placeholder="Type to filter...")

            rows = []
            for r in records:
                if search and search.lower() not in (r.patient_name or "").lower() \
                              and search.lower() not in (r.patient_id or "").lower():
                    continue
                rows.append({
                    "Analysis ID":  r.analysis_id,
                    "Patient":      r.patient_name or "—",
                    "Age":          str(r.patient_age) if r.patient_age is not None else "—",
                    "Date":         r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "—",
                    "RBC (/µL)":    f"{r.cal_rbc:,}",
                    "WBC (/µL)":    f"{r.cal_wbc:,}",
                    "Platelet":     f"{r.cal_platelet:,}",
                    "Confidence":   f"{r.avg_confidence:.0%}",
                    "Result":       r.overall_result,
                })

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(
                    df.style.apply(
                        lambda col: ["color:#f85149" if v=="Abnormal" else "color:#3fb950"
                                     for v in col] if col.name == "Result" else [""]*len(col),
                        axis=0
                    ),
                    hide_index=True, use_container_width=True, height=400,
                )

                # Trend chart if enough records
                if len(records) >= 3:
                    st.markdown('<div class="section-title">WBC Trend</div>', unsafe_allow_html=True)
                    trend_df = pd.DataFrame([{
                        "Date": r.created_at, "WBC /µL": r.cal_wbc, "RBC /µL": r.cal_rbc
                    } for r in records if r.created_at]).sort_values("Date")
                    fig2 = px.line(trend_df, x="Date", y=["WBC /µL", "RBC /µL"],
                                   color_discrete_map={"WBC /µL": "#3b82f6", "RBC /µL": "#ef4444"})
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                       plot_bgcolor="rgba(0,0,0,0)",
                                       font_color="#e6edf3", height=260,
                                       margin=dict(t=10,b=30,l=10,r=10))
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No records match your search.")

    except Exception as e:
        st.error(f"Database error: {e}")
        st.code(str(e))


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ## AI Blood Report Analyzer — Phase 2

    ### Pipeline
    | Step | Module | Technology |
    |---|---|---|
    | Preprocessing | `src/preprocess.py` | OpenCV · CLAHE · Gaussian blur |
    | Cell Detection | `src/detect.py` | YOLOv8n (keremberke BCCD, 5.9 MB) |
    | WBC Classification | `src/classify_wbc.py` | sklearn SVM+RF ensemble |
    | Normal Range Check | `src/normal_ranges.py` | WHO reference ranges |
    | Report Generation | `src/report_generator.py` | FPDF2 · Matplotlib |
    | Patient Records | `database/` | SQLite · SQLAlchemy |
    | Batch Analysis | `src/batch_analyze.py` | Multi-image averaging |

    ### Dataset
    - **BCCD**: 364 blood smear images · 372 WBC crops extracted
    - **WBC Classifier training**: sklearn ensemble (SVM + RandomForest) on 297 crops
    - **Val accuracy**: ~72% (pseudo-labeled; will improve with real WBC subtype labels)

    ### Phase 3 Roadmap
    - Docker containerisation for lab deployment
    - Upload real WBC subtype dataset (e.g. Kaggle Blood-Cell-Images) for retraining
    - Replace sklearn with EfficientNet-B0 (requires GPU or cloud)
    - Multi-slide stitching for full slide CBC estimation
    - HL7/FHIR integration for EMR export
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**BCCD images:** 364  \n**WBC crops:** 372  \n**YOLOv8 classes:** Platelet · RBC · WBC")
    with col2:
        wbc_exists = project_path("models", "resnet50_wbc", "wbc_classifier_sklearn.pth").exists()
        yolo_exists = project_path("models", "yolov8_blood", "best.pt").exists()
        st.success(f"YOLOv8: {'✓ Loaded' if yolo_exists else '✗ Missing'}  \nWBC Model: {'✓ sklearn' if wbc_exists else '⚠ Placeholder'}")
