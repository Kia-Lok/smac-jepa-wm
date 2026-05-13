from __future__ import annotations

from pathlib import Path


def write_svg_line_plot(
    rows: list[dict[str, float | int]],
    x_key: str,
    y_key: str,
    title: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    if not rows:
        out_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n")
        return

    values = [(float(row[x_key]), float(row[y_key])) for row in rows]
    xs = [x for x, _ in values]
    ys = [y for _, y in values]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    width, height = 900, 320
    left, right, top, bottom = 70, 26, 42, 54
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + ((x - min_x) / (max_x - min_x)) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - ((y - min_y) / (max_y - min_y)) * plot_h

    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
    circles = "\n".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3"><title>{x_key}={x:g}, {y_key}={y:.6g}</title></circle>'
        for x, y in values
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1f2933; }}
    .axis {{ stroke: #7b8794; stroke-width: 1; }}
    .series {{ fill: none; stroke: #2563eb; stroke-width: 3; }}
    circle {{ fill: #2563eb; }}
    .tick {{ fill: #5f6b76; font-size: 12px; }}
  </style>
  <title>{title}</title>
  <text x="{left}" y="24" font-size="18" font-weight="700">{title}</text>
  <line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" />
  <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" />
  <text class="tick" x="{left}" y="{height - 16}">{min_x:g}</text>
  <text class="tick" x="{left + plot_w - 22}" y="{height - 16}">{max_x:g}</text>
  <text class="tick" x="10" y="{top + plot_h}">{min_y:.4g}</text>
  <text class="tick" x="10" y="{top + 4}">{max_y:.4g}</text>
  <polyline class="series" points="{points}" />
  {circles}
</svg>
"""
    out_path.write_text(svg)

