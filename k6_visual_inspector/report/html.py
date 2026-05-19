"""HTML report generation."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import List

from PIL import Image

from ..models import Cluster, ScreenshotItem


def _image_to_base64_data_uri(path: Path, max_width: int = 360) -> str:
    image = Image.open(path).convert("RGB")
    w, h = image.size

    if w > max_width:
        new_h = int(h * (max_width / w))
        image = image.resize((max_width, new_h))

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def generate_html_report(
    items: List[ScreenshotItem],
    clusters: List[Cluster],
    out_dir: Path,
    visual_weight: float,
    text_weight: float,
    layout_weight: float,
    rule_weight: float,
    threshold: float,
    template_threshold: float,
    final_threshold: float,
    single_stage: bool,
) -> None:
    """Write report.html to *out_dir*."""
    rows: List[str] = []

    for cluster in clusters:
        rep = items[cluster.representative_index]
        img_uri = _image_to_base64_data_uri(Path(rep.path))
        overlay_path = out_dir / "overlays" / f"cluster-{cluster.cluster_id:03d}.jpg"
        overlay_uri = _image_to_base64_data_uri(overlay_path) if overlay_path.exists() else ""

        token_html = ", ".join(html.escape(t) for t in cluster.common_tokens[:20])
        sample_text = html.escape(rep.normalized_text[:1200])
        central_text = html.escape(rep.central_ocr_text[:1200])
        variants_html = "\n".join(
            f"<li><code>{html.escape(v)}</code></li>"
            for v in cluster.text_variants
        )

        rows.append(
            f"""
            <section class="cluster severity-{html.escape(cluster.severity)}">
              <div class="cluster-header">
                <div>
                  <h2>Cluster {cluster.cluster_id:03d}: {html.escape(cluster.cluster_name)}</h2>
                  <div class="subtle">
                    visual={html.escape(cluster.visual_class)} |
                    text={html.escape(cluster.text_class)} |
                    severity={html.escape(cluster.severity)}
                  </div>
                </div>
                <div class="count">{cluster.count} screenshots</div>
              </div>

              <div class="metrics">
                <span>combined: {cluster.avg_combined_similarity:.3f}</span>
                <span>visual: {cluster.avg_visual_similarity:.3f}</span>
                <span>text: {cluster.avg_text_similarity:.3f}</span>
                <span>layout: {cluster.avg_layout_similarity:.3f}</span>
                <span>rule: {cluster.avg_rule_similarity:.3f}</span>
              </div>

              <div class="images">
                <div>
                  <h3>Representative</h3>
                  <img src="{img_uri}" />
                </div>
                <div>
                  <h3>Detected layout</h3>
                  <img src="{overlay_uri}" />
                </div>
              </div>

              <div class="details">
                <p><b>Representative file:</b> {html.escape(rep.filename)}</p>
                <p><b>Common OCR tokens:</b> {token_html}</p>

                <h3>Normalized OCR sample</h3>
                <pre>{sample_text}</pre>

                <h3>Central OCR sample</h3>
                <pre>{central_text}</pre>

                <h3>Text variants</h3>
                <ol>{variants_html}</ol>
              </div>
            </section>
            """
        )

    mode_text = "single-stage" if single_stage else "two-stage"

    doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Screenshot Cluster Report</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f9fafb;
    }}
    h1 {{
      margin-bottom: 4px;
    }}
    .summary {{
      margin-bottom: 24px;
      padding: 16px;
      background: white;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
    }}
    .cluster {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    .cluster-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .cluster-header h2 {{
      margin: 0;
    }}
    .subtle {{
      color: #6b7280;
      margin-top: 4px;
      font-size: 14px;
    }}
    .count {{
      font-weight: 700;
      background: #eef2ff;
      padding: 6px 10px;
      border-radius: 999px;
      white-space: nowrap;
    }}
    .metrics {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 12px 0;
    }}
    .metrics span {{
      background: #f3f4f6;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 13px;
    }}
    .images {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-top: 12px;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #f9fafb;
      border-radius: 8px;
      padding: 12px;
      max-height: 260px;
      overflow: auto;
    }}
    code {{
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .severity-high {{
      border-left: 6px solid #ef4444;
    }}
    .severity-critical {{
      border-left: 6px solid #7f1d1d;
    }}
    .severity-medium {{
      border-left: 6px solid #f59e0b;
    }}
    .severity-low {{
      border-left: 6px solid #22c55e;
    }}
  </style>
</head>
<body>
  <h1>Screenshot Cluster Report</h1>

  <div class="summary">
    <p><b>Total screenshots:</b> {len(items)}</p>
    <p><b>Total clusters:</b> {len(clusters)}</p>
    <p><b>Mode:</b> {html.escape(mode_text)}</p>
    <p><b>Single-stage threshold:</b> {threshold:.3f}</p>
    <p><b>Template threshold:</b> {template_threshold:.3f}</p>
    <p><b>Final threshold:</b> {final_threshold:.3f}</p>
    <p><b>Weights:</b> visual={visual_weight}, text={text_weight}, layout={layout_weight}, rule={rule_weight}</p>
  </div>

  {''.join(rows)}
</body>
</html>
"""

    (out_dir / "report.html").write_text(doc, encoding="utf-8")
