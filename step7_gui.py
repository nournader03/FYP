from __future__ import annotations
import sys, tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2

from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import gradio as gr

# Config
IMG_SIZE = 224
FCD_CLASS = 1                      # label_int: FCD=1, Control=0
MODEL_FOLDER = "convnext_tiny"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_TFM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def find_root() -> Path:
    for s in [Path.cwd(), Path(__file__).resolve().parent]:
        for d in [s, *s.parents]:
            if (d / "models").is_dir() and (d / "data").is_dir():
                return d
    return Path.cwd()


ROOT = find_root()
CKPT_DIR = ROOT / "models" / MODEL_FOLDER


# --------------------------------------------------------------------------- #
# Model (ConvNeXt-Tiny, torchvision, 2-logit head) - load all folds once
# --------------------------------------------------------------------------- #
def build_convnext_tiny() -> nn.Module:
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    return m


def load_fold_models():
    mods = []
    for k in range(1, 6):
        ck = CKPT_DIR / f"fold{k}_best.pt"
        if not ck.exists():
            continue
        m = build_convnext_tiny()
        state = torch.load(str(ck), map_location=DEVICE)
        m.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        m.eval().to(DEVICE)
        mods.append(m)
    return mods


def compute_thresholds():
    """Balanced (Youden) and screening (sens>=0.95) thresholds from OOF probs."""
    pf = CKPT_DIR / "oof_probs.npy"
    lf = CKPT_DIR / "oof_labels.npy"
    if not (pf.exists() and lf.exists()):
        return {"balanced": 0.5, "screening": 0.5}
    from sklearn.metrics import roc_curve
    p = np.load(pf).ravel(); y = np.load(lf).astype(int).ravel()
    fpr, tpr, thr = roc_curve(y, p)
    bal = float(thr[int(np.argmax(tpr - fpr))])
    mask = tpr >= 0.95
    scr = float(thr[int(np.argmax(mask))]) if mask.any() else float(thr[-1])
    return {"balanced": bal, "screening": scr}


print(f"Loading ConvNeXt-Tiny from {CKPT_DIR} on {DEVICE} ...")
FOLD_MODELS = load_fold_models()
THRESHOLDS = compute_thresholds()
print(f"  {len(FOLD_MODELS)} fold checkpoint(s) loaded. "
      f"Thresholds: balanced={THRESHOLDS['balanced']:.3f}, "
      f"screening={THRESHOLDS['screening']:.3f}")


# --------------------------------------------------------------------------- #
# Preprocessing - reuse step2_preprocess helpers for identical behaviour
# --------------------------------------------------------------------------- #
def preprocess_upload(nii_path: str, cache_dir: Path):
    """Replicates step2.process_subject on an uploaded volume. Returns list of
    (slice_idx, png_path) using the SAME helpers/PNG creation as training."""
    import nibabel as nib
    try:
        sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
        import step2_preprocess as s2
    except Exception as e:
        raise RuntimeError(
            "Could not import step2_preprocess.py - GUI in the same "
            f"project so preprocessing matches training. ({e})")

    use_hdbet = s2.check_hdbet()
    img = nib.as_closest_canonical(nib.load(str(nii_path)))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.squeeze(data)
    if data.shape[1] < data.shape[2] and data.shape[1] < data.shape[0]:
        data = np.transpose(data, (0, 2, 1))
    elif data.shape[0] < data.shape[2] and data.shape[0] < data.shape[1]:
        data = np.transpose(data, (1, 2, 0))

    canonical = nib.Nifti1Image(data, img.affine, img.header)
    stripped = s2.apply_skull_strip(canonical, Path(nii_path), cache_dir, use_hdbet)
    normed = s2.percentile_normalize(stripped)

    out = []
    for idx in s2.select_slices(normed):
        sl = normed[:, :, idx].T
        if not s2.is_informative_slice(sl):
            continue
        png = cache_dir / f"slice{idx:04d}.png"
        s2.save_slice_png(sl, png)
        out.append((idx, str(png)))
    return out


def load_png(png_path: str):
    arr = np.array(Image.open(png_path)).astype(np.float32) / 65535.0
    arr = np.clip(arr, 0, 1)
    pil = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")
    rgb = np.asarray(transforms.CenterCrop(IMG_SIZE)(
        transforms.Resize(IMG_SIZE)(pil))).astype(np.float32) / 255.0
    return VAL_TFM(pil).unsqueeze(0), rgb


# --------------------------------------------------------------------------- #
# Inference + Grad-CAM
# --------------------------------------------------------------------------- #
@torch.no_grad()
def infer(png_paths):
    tensors = torch.cat([load_png(p)[0] for p in png_paths]).to(DEVICE)
    fold_probs = []
    for m in FOLD_MODELS:
        fold_probs.append(torch.softmax(m(tensors), dim=1)[:, FCD_CLASS])
    slice_probs = torch.stack(fold_probs).mean(0)          # avg folds, per slice
    subject_prob = float(slice_probs.mean().item())         # avg slices
    return subject_prob, slice_probs.cpu().numpy()


def gradcam_overlay(png_path):
    """Grad-CAM for the most suspicious slice, using the first fold model."""
    model = FOLD_MODELS[0]
    tensor, rgb = load_png(png_path)
    tensor = tensor.to(DEVICE)
    cam_engine = GradCAM(model=model, target_layers=[model.features[-1]])
    cam = cam_engine(input_tensor=tensor,
                     targets=[ClassifierOutputTarget(FCD_CLASS)])[0]
    return show_cam_on_image(rgb, cam, use_rgb=True)


def predict(nii_file, threshold_mode):
    if nii_file is None:
        return "Please upload a FLAIR NIfTI file.", None
    thr = THRESHOLDS["screening" if threshold_mode.startswith("Screening")
                     else "balanced"]
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        try:
            slices = preprocess_upload(nii_file, cache)
        except Exception as e:
            return f"Preprocessing failed: {e}", None
        if not slices:
            return "No usable brain slices found in the volume.", None

        png_paths = [p for _, p in slices]
        subject_prob, slice_probs = infer(png_paths)
        label = "FCD" if subject_prob >= thr else "Normal"
        confidence = subject_prob if label == "FCD" else 1 - subject_prob

        top_png = png_paths[int(np.argmax(slice_probs))]
        overlay = gradcam_overlay(top_png)

    md = (f"## Prediction: **{label}**\n\n"
          f"- P(FCD) = **{subject_prob*100:.1f}%**  (confidence in call: "
          f"{confidence*100:.1f}%)\n"
          f"- Operating point: **{threshold_mode}** (threshold = {thr:.3f})\n"
          f"- Slices analysed: {len(png_paths)}\n\n"
          f"*Grad-CAM shows the most FCD-suspicious slice. Note: in validation "
          f"the model's attention did not reliably localise to lesions, so the "
          f"heatmap is indicative of model focus, not a lesion marker.*")
    return md, overlay


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
DISCLAIMER = ("### FCD Detection - Research Prototype\n"
              "**Not for diagnostic use.** This tool is a final-year research "
              "prototype (ConvNeXt-Tiny, ds004199). Predictions are not "
              "validated for clinical decision-making.")


def build_ui():
    with gr.Blocks(title="FCD Detection (ConvNeXt-Tiny)") as demo:
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=1):
                nii = gr.File(label="Upload FLAIR (.nii or .nii.gz)",
                              file_types=[".nii", ".gz"], type="filepath")
                mode = gr.Radio(["Balanced (max Youden's J)",
                                 "Screening (sensitivity >= 0.95)"],
                                value="Balanced (max Youden's J)",
                                label="Operating point")
                btn = gr.Button("Analyse", variant="primary")
                gr.Markdown("*Processing includes skull-stripping and may take "
                            "up to a few minutes per scan.*")
            with gr.Column(scale=1):
                out_md = gr.Markdown(label="Result")
                out_img = gr.Image(label="Grad-CAM (most suspicious slice)")
        btn.click(predict, inputs=[nii, mode], outputs=[out_md, out_img])
    return demo


if __name__ == "__main__":
    if not FOLD_MODELS:
        print(f"ERROR: no ConvNeXt-Tiny checkpoints found in {CKPT_DIR}")
        sys.exit(1)
    build_ui().launch()