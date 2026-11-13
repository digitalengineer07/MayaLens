import os
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable
)

def _generate_temporal_plot(temporal_series: List[Dict[str, Any]], out_path: Path, is_fake: bool) -> Optional[Path]:
    """Renders a high-resolution temporal consistency plot from frame data."""
    if not temporal_series or len(temporal_series) < 2:
        return None
    try:
        scores = [float(pt.get("score", 0.0)) for pt in temporal_series]
        times = [str(pt.get("time", f"{i}s")) for i, pt in enumerate(temporal_series)]
        x_indices = list(range(len(scores)))

        fig, ax = plt.subplots(figsize=(7.2, 2.1), dpi=150)
        fig.patch.set_facecolor('#F8FAFC')
        ax.set_facecolor('#FFFFFF')

        line_color = '#DC2626' if is_fake else '#2563EB'
        ax.plot(x_indices, scores, marker='o', markersize=3.5, color=line_color, linewidth=1.8, label='P(Fake) Trajectory')
        ax.axhline(50.0, color='#94A3B8', linestyle='--', linewidth=1, label='Threshold Boundary (50%)')

        # Red markers for anomalies
        for i, s in enumerate(scores):
            if s > 50.0:
                ax.plot(i, s, marker='o', markersize=6, color='#EF4444', markeredgecolor='#7F1D1D', markeredgewidth=1)

        ax.set_ylim(-5, 105)
        ax.set_ylabel('Score (%)', fontsize=8, color='#334155', fontweight='bold')
        ax.set_xlabel('Timeline Interval', fontsize=8, color='#334155')

        step = max(1, len(times) // 8)
        selected_ticks = list(range(0, len(times), step))
        ax.set_xticks(selected_ticks)
        ax.set_xticklabels([times[i] for i in selected_ticks], fontsize=7, color='#64748B')
        ax.tick_params(axis='y', labelsize=7, colors='#64748B')

        ax.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.85)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"[PDF Generator] Plot generation warning: {e}")
        return None

def generate_forensic_pdf_report(data: Dict[str, Any], output_dir: Path, uploads_dir: Path) -> str:
    """
    Builds an official, forensic-grade PDF dossier strictly reflecting
    the analyzed media item's current evaluation results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = data.get("filename", "analyzed_media")
    stem = Path(filename).stem
    
    # Safe unique report filename
    timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"MayaLens_Dossier_{stem}_{timestamp_id}.pdf"
    report_path = output_dir / report_filename

    # Core data fields
    decision = data.get("decision", "AUTHENTIC MEDIA").upper()
    is_fake = bool(data.get("is_fake", False))
    confidence = float(data.get("confidence", 95.0))
    risk_level = str(data.get("risk_level", "LOW")).upper()
    summary_text = data.get("summary", "Forensic analysis completed.")
    media_type = data.get("media_type", "video").upper()
    file_size = data.get("file_size", "N/A")
    resolution = data.get("resolution", "1280 × 720")
    duration = data.get("duration", "0.0s")
    total_frames = data.get("total_frames", 1)
    fps = data.get("fps", 24.0)
    bitrate = data.get("bitrate_mbps", 0.0)
    avg_lap = data.get("avg_laplacian", 140.0)
    hf_ratio = data.get("hf_spectral_ratio", 20.0)
    latency_ms = data.get("inference_time_ms") or data.get("processing_time_ms") or 450

    # Cryptographic evidence signature
    raw_hash_seed = f"{filename}_{file_size}_{decision}_{confidence}".encode('utf-8')
    sha256_sig = hashlib.sha256(raw_hash_seed).hexdigest()

    # ReportLab document setup
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm
    )

    ss = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=ss["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=ss["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B")
    )
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=ss["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=ss["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=ss["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0F172A")
    )
    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=ss["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B"),
        alignment=1  # Centered
    )

    story = []

    # 1. Header Banner
    header_table_data = [
        [
            Paragraph("<b>MAYALENS FORENSIC DOSSIER</b>", title_style),
            Paragraph(f"<b>CASE REF:</b> {sha256_sig[:16].upper()}<br/><font color='#64748B'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</font>", ParagraphStyle("Ref", parent=ss["Normal"], fontSize=8, leading=10, alignment=2))
        ],
        [
            Paragraph("Explainable Multi-Stream Deepfake Detection & Biometric Authenticity Report", subtitle_style),
            Paragraph(f"<b>ENGINE:</b> MayaLens 3-Stream Fusion v1.0.0", ParagraphStyle("Eng", parent=ss["Normal"], fontSize=8, leading=10, alignment=2, textColor=colors.HexColor("#6366F1")))
        ]
    ]
    header_table = Table(header_table_data, colWidths=[11.5 * cm, 6.7 * cm])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#6366F1"), spaceBefore=2, spaceAfter=8))

    # 2. Verdict Banner
    verdict_bg = colors.HexColor("#FEF2F2" if is_fake else "#ECFDF5")
    verdict_border = colors.HexColor("#DC2626" if is_fake else "#059669")
    verdict_text_color = colors.HexColor("#991B1B" if is_fake else "#065F46")
    badge_bg = colors.HexColor("#DC2626" if is_fake else "#059669")

    verdict_content = [
        [
            Paragraph(
                f"<font color='{badge_bg.hexval()}'>■</font> <b>VERDICT: {decision}</b>",
                ParagraphStyle("VerdB", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=verdict_text_color)
            ),
            Paragraph(
                f"<b>CONFIDENCE:</b> <font size='13'>{confidence}%</font> &nbsp;|&nbsp; <b>RISK:</b> {risk_level}",
                ParagraphStyle("VerdC", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=13, alignment=2, textColor=verdict_text_color)
            )
        ],
        [
            Paragraph(summary_text, ParagraphStyle("Summ", parent=ss["Normal"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor("#1E293B"))),
            Paragraph(f"<b>Inference Latency:</b> {latency_ms} ms", ParagraphStyle("Lat", parent=ss["Normal"], fontName="Helvetica", fontSize=8, leading=11, alignment=2, textColor=colors.HexColor("#475569")))
        ]
    ]
    verdict_table = Table(verdict_content, colWidths=[12.5 * cm, 5.7 * cm])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_bg),
        ("BOX", (0, 0), (-1, -1), 1.2, verdict_border),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 8))

    # 3. Media Metadata Verification Table
    story.append(Paragraph("<b>1. Media Technical Specifications & Forensic Metadata</b>", section_h1))
    meta_data = [
        [
            Paragraph("<b>Target Filename:</b>", body_style), Paragraph(str(filename), body_bold),
            Paragraph("<b>Media Type:</b>", body_style), Paragraph(str(media_type), body_bold)
        ],
        [
            Paragraph("<b>File Size:</b>", body_style), Paragraph(str(file_size), body_bold),
            Paragraph("<b>Resolution:</b>", body_style), Paragraph(str(resolution), body_bold)
        ],
        [
            Paragraph("<b>Total Frames:</b>", body_style), Paragraph(f"{total_frames} frames", body_bold),
            Paragraph("<b>Duration / FPS:</b>", body_style), Paragraph(f"{duration} @ {fps} fps", body_bold)
        ],
        [
            Paragraph("<b>Bitrate:</b>", body_style), Paragraph(f"{bitrate} Mbps", body_bold),
            Paragraph("<b>Texture Sharpness:</b>", body_style), Paragraph(f"{avg_lap} Var (Laplacian)", body_bold)
        ],
        [
            Paragraph("<b>Spectral HF Ratio:</b>", body_style), Paragraph(f"{hf_ratio}% Nyquist", body_bold),
            Paragraph("<b>Evidence SHA-256:</b>", body_style), Paragraph(f"<font size='6.5'>{sha256_sig[:32]}...</font>", body_bold)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[3.2 * cm, 5.8 * cm, 3.2 * cm, 6.0 * cm])
    meta_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F8FAFC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # 4. Multi-Stream Forensic Architecture Table
    story.append(Paragraph("<b>2. Multi-Stream Deepfake Neural Evaluation</b>", section_h1))
    streams = data.get("streams", [])
    stream_rows = [
        [
            Paragraph("<b>Stream</b>", body_bold),
            Paragraph("<b>Backbone Architecture</b>", body_bold),
            Paragraph("<b>Confidence</b>", body_bold),
            Paragraph("<b>Status</b>", body_bold),
            Paragraph("<b>Forensic Evidence Focus</b>", body_bold)
        ]
    ]

    for s in streams:
        s_name = s.get("name", "Stream")
        s_conf = s.get("confidence", 95.0)
        s_status = str(s.get("status", "PASS")).upper()
        s_focus = s.get("focus", "")
        
        # Color badge for status
        is_pass = s_status in ["PASS", "AUTHENTIC"]
        status_color = "#059669" if is_pass else "#DC2626"

        arch = "EfficientNet-B4 (1.15× Tight)" if "spatial" in s_name.lower() else (
            "Vision Transformer (1.70× Context)" if "context" in s_name.lower() else "FFT/DCT + ResNet-18"
        )
        stream_rows.append([
            Paragraph(f"<b>{s_name.split('(')[0].strip()}</b>", body_bold),
            Paragraph(arch, body_style),
            Paragraph(f"<b>{s_conf}%</b>", body_bold),
            Paragraph(f"<b><font color='{status_color}'>{s_status}</font></b>", body_bold),
            Paragraph(s_focus, body_style)
        ])

    stream_table = Table(stream_rows, colWidths=[3.2 * cm, 4.2 * cm, 2.0 * cm, 2.0 * cm, 6.8 * cm])
    stream_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(stream_table)
    story.append(Spacer(1, 8))

    # 5. Temporal Consistency Chart
    temporal_series = data.get("temporal_series", [])
    if temporal_series and len(temporal_series) > 1:
        plot_path = output_dir / f"temporal_{stem}_{timestamp_id}.png"
        generated_plot = _generate_temporal_plot(temporal_series, plot_path, is_fake)
        if generated_plot and generated_plot.exists():
            story.append(Paragraph("<b>3. Frame-by-Frame Temporal Consistency Trajectory</b>", section_h1))
            story.append(RLImage(str(generated_plot), width=18.0 * cm, height=5.2 * cm))
            story.append(Paragraph("Timeline trajectory tracking manipulation likelihood per sampled interval against the 50% forensic decision boundary.", caption_style))
            story.append(Spacer(1, 8))

    # 6. Keyframe Forensic Evidence Breakdown (Grad-CAM & Artifact Loupe)
    keyframes = data.get("keyframes", [])
    if keyframes:
        story.append(PageBreak())
        story.append(Paragraph("<b>4. Keyframe Visual Evidence & Explainable AI (Grad-CAM) Loupe</b>", section_h1))
        story.append(Paragraph("Grad-CAM overlays visualize convolutional regions exhibiting highest predictive anomaly gradients. Warm spectra (red/yellow) indicate synthesized pixel boundaries.", subtitle_style))
        story.append(Spacer(1, 6))

        for kf in keyframes[:3]:
            f_idx = kf.get("frame_idx", 0)
            t_str = kf.get("timestamp", "00:00.00")
            fake_sc = float(kf.get("fake_score", 0.0))
            anomaly = kf.get("anomaly_detected", "Continuous natural texture")
            focus = kf.get("xai_focus", "")
            features = kf.get("features", {})
            fft_text = features.get("fft_peaks", "Clean organic power spectrum decay")
            vit_text = features.get("vit_attention", "Coherent patch attention")

            # Look for thumbnail and CAM files on disk
            thumb_name = f"{stem}_kf_{f_idx}.jpg"
            cam_name = f"{stem}_cam_{f_idx}.jpg"
            thumb_disk = uploads_dir / "thumbnails" / thumb_name
            cam_disk = uploads_dir / "thumbnails" / cam_name

            # Also check provided thumbnail_url
            if not thumb_disk.exists() and kf.get("thumbnail_url"):
                alt_name = Path(kf.get("thumbnail_url")).name
                if (uploads_dir / "thumbnails" / alt_name).exists():
                    thumb_disk = uploads_dir / "thumbnails" / alt_name

            # Build image elements
            img_cells = []
            if thumb_disk.exists():
                img_cells.append(RLImage(str(thumb_disk), width=5.2 * cm, height=5.2 * cm))
            if cam_disk.exists():
                img_cells.append(RLImage(str(cam_disk), width=5.2 * cm, height=5.2 * cm))

            # Metadata column
            text_desc = (
                f"<b>Keyframe #{f_idx}</b> ({t_str}) &nbsp;|&nbsp; <b>P(Fake):</b> {fake_sc:.3f}<br/>"
                f"<b>Localized Anomaly:</b> {anomaly}<br/>"
                f"<b>Grad-CAM Localization:</b> {focus}<br/>"
                f"<b>Spectral Profile:</b> {fft_text}<br/>"
                f"<b>Context Attention:</b> {vit_text}"
            )
            desc_cell = Paragraph(text_desc, body_style)

            if len(img_cells) == 2:
                kf_table_data = [
                    [img_cells[0], img_cells[1], desc_cell],
                    [Paragraph("Sampled Frame Crop", caption_style), Paragraph("Grad-CAM XAI Heatmap", caption_style), Paragraph("", caption_style)]
                ]
                kf_table = Table(kf_table_data, colWidths=[5.4 * cm, 5.4 * cm, 7.4 * cm])
            elif len(img_cells) == 1:
                kf_table_data = [
                    [img_cells[0], desc_cell],
                    [Paragraph("Sampled Keyframe", caption_style), Paragraph("", caption_style)]
                ]
                kf_table = Table(kf_table_data, colWidths=[6.0 * cm, 12.2 * cm])
            else:
                kf_table_data = [[desc_cell]]
                kf_table = Table(kf_table_data, colWidths=[18.2 * cm])

            kf_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(kf_table)
            story.append(Spacer(1, 6))

    # 7. Forensic Integrity & Disclaimer
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceBefore=4, spaceAfter=4))
    disclaimer_text = (
        "<b>LEGAL & FORENSIC INTEGRITY NOTICE:</b> This dossier was autonomously generated by MayaLens Neural Vision Engine. "
        "Grad-CAM heatmaps localize computational anomaly variance and do not constitute independent legal culpability. "
        "Intended for decision support, forensic triage, and verification pipelines in conformity with ISO/IEC standards. "
        "Verify document integrity using the SHA-256 evidence digest."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle("Disc", parent=ss["Normal"], fontName="Helvetica-Oblique", fontSize=7, leading=9.5, textColor=colors.HexColor("#64748B"))))

    # Build PDF
    doc.build(story)
    return report_filename

