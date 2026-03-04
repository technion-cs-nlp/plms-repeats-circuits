"""
Analyze counterfactual experiment results: compare main vs baseline methods.
Export run_analysis() for use from CLI or notebook.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plms_repeats_circuits.utils.counterfactuals_config import identify_result_file

try:
    import kaleido
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
except ImportError:
    pass


def discover_files(input_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Discover counterfactual result files; separate main vs baseline."""
    main_files = {}
    baseline_files = {}
    for csv_file in sorted(input_dir.rglob("*.csv")):
        display_name, kind = identify_result_file(csv_file.stem)
        if display_name is None:
            continue
        if kind == "baseline":
            baseline_files[display_name] = csv_file
        else:
            main_files[display_name] = csv_file
    return main_files, baseline_files


def load_and_compute_metrics(file_path: Path) -> Tuple[float, float, float]:
    """Load CSV, compute mean prediction change and mean prob diff."""
    df = pd.read_csv(file_path)
    mean_pred = df["is_corrupt_changed_argmax"].mean()
    prob_diff = df["true_token_prob_before"] - df["true_token_prob_after"]
    mean_prob = prob_diff.mean()
    std_prob = prob_diff.std()
    return mean_pred, mean_prob, std_prob


def create_comparison_plot(
    main_files: Dict[str, Path],
    baseline_files: Dict[str, Path],
    output_path: Path,
    sort_by: str = "prediction_change",
):
    """Create grouped bar chart (Real vs Baseline Counterfactual)."""
    main_names, main_pred, main_prob = [], [], []
    for name, path in main_files.items():
        try:
            pred, prob, _ = load_and_compute_metrics(path)
            main_names.append(name)
            main_pred.append(pred)
            main_prob.append(prob)
        except Exception as e:
            print(f"Error {name}: {e}")
    base_names, base_pred, base_prob = [], [], []
    for name, path in baseline_files.items():
        try:
            pred, prob, _ = load_and_compute_metrics(path)
            base_names.append(name)
            base_pred.append(pred)
            base_prob.append(prob)
        except Exception as e:
            print(f"Error {name}: {e}")

    if main_names:
        idx = 1 if sort_by == "prediction_change" else 2
        sorted_main = sorted(zip(main_names, main_pred, main_prob), key=lambda x: x[idx], reverse=True)
        main_names, main_pred, main_prob = [list(x) for x in zip(*sorted_main)]
    if base_names and main_names:
        base_dict = {n: (p, pr) for n, p, pr in zip(base_names, base_pred, base_prob)}
        base_names = [n for n in main_names if n in base_dict] + [n for n in base_names if n not in main_names]
        base_pred = [base_dict[n][0] for n in base_names]
        base_prob = [base_dict[n][1] for n in base_names]
    elif base_names:
        idx = 1 if sort_by == "prediction_change" else 2
        sorted_base = sorted(zip(base_names, base_pred, base_prob), key=lambda x: x[idx], reverse=True)
        base_names, base_pred, base_prob = [list(x) for x in zip(*sorted_base)]

    n_cols = (1 if main_names else 0) + (1 if base_names else 0)
    if n_cols == 0:
        print("No data to plot")
        return

    titles = []
    if main_names:
        titles.append("Real Counterfactual")
    if base_names:
        titles.append("Baseline Counterfactual")

    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles, horizontal_spacing=0.12)
    color_pred = "#045a8d"
    color_prob = "#a6bddb"

    col = 1
    if main_names:
        fig.add_trace(
            go.Bar(
                x=main_names,
                y=main_pred,
                name="Top-1 Prediction Change Fraction",
                marker=dict(color=color_pred, line=dict(color="black", width=0.8)),
                text=[f"{v:.2f}" for v in main_pred],
                textposition="outside",
                textfont=dict(size=16),
                showlegend=True,
                legendgroup="m",
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Bar(
                x=main_names,
                y=main_prob,
                name="Probability Difference of Correct Token<br>(Before - After Corruption)",
                marker=dict(color=color_prob, line=dict(color="black", width=0.8)),
                text=[f"{v:.2f}" for v in main_prob],
                textposition="outside",
                textfont=dict(size=16),
                showlegend=True,
                legendgroup="m",
            ),
            row=1, col=col,
        )
        col += 1
    if base_names:
        fig.add_trace(
            go.Bar(
                x=base_names,
                y=base_pred,
                marker=dict(color=color_pred, line=dict(color="black", width=0.8)),
                text=[f"{v:.2f}" for v in base_pred],
                textposition="outside",
                textfont=dict(size=16),
                showlegend=False,
                legendgroup="m",
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Bar(
                x=base_names,
                y=base_prob,
                marker=dict(color=color_prob, line=dict(color="black", width=0.8)),
                text=[f"{v:.2f}" for v in base_prob],
                textposition="outside",
                textfont=dict(size=16),
                showlegend=False,
                legendgroup="m",
            ),
            row=1, col=col,
        )

    w = 600 * n_cols + 100
    h = 500
    fig.update_layout(
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            orientation="v",
            font=dict(size=16),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
        ),
        height=h,
        width=w,
        plot_bgcolor="white",
        margin=dict(b=150, l=80, r=120, t=80),
    )
    for annotation in fig.layout.annotations:
        annotation.font.size = 16
    fig.update_xaxes(
        title="Corruption Method",
        title_font=dict(size=16),
        tickangle=45,
        tickfont=dict(size=16),
        showgrid=False,
        showline=True,
        linewidth=1.2,
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        range=[0, 1.02],
        showgrid=True,
        gridwidth=0.3,
        gridcolor="lightgray",
        showline=True,
        linewidth=1.2,
        linecolor="black",
        mirror=True,
        tickfont=dict(size=16),
    )
    fig.update_yaxes(
        title="Fraction Changed / Mean Probability Drop",
        title_font=dict(size=16),
        row=1, col=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in [".png", ".pdf"]:
        out = output_path.with_suffix(ext)
        try:
            fig.write_image(str(out), width=w, height=h)
            print(f"Saved {out}")
        except Exception as e:
            print(f"Could not save {ext}: {e}")


def infer_output_filename(input_dir: Path) -> str:
    """Infer output filename from input path."""
    s = str(input_dir).lower()
    model = "esmc" if "esm-c" in s or "esmc" in s else ("esm3" if "esm3" in s else "unknown")
    repeat = "random" if "random" in s else ("approximate" if "similar" in s or "similiar" in s else "identical")
    return f"{model}_{repeat}_counterfactual_comparison.pdf"


def run_analysis(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    sort_by: str = "prediction_change",
) -> None:
    """
    Run counterfactual analysis: discover files, create comparison plot.
    Callable from notebook or CLI.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    main_files, baseline_files = discover_files(input_dir)
    total = len(main_files) + len(baseline_files)
    if total == 0:
        print("No counterfactual result files found")
        return
    print(f"Found {len(main_files)} main, {len(baseline_files)} baseline")
    filename = infer_output_filename(input_dir)
    output_path = output_dir / filename
    create_comparison_plot(main_files, baseline_files, output_path, sort_by=sort_by)


def main(args=None):
    parser = argparse.ArgumentParser(description="Compare counterfactual methods and generate plots")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--sort_by", choices=["prediction_change", "prob_diff"], default="prediction_change")
    parsed = parser.parse_args(args)
    run_analysis(parsed.input_dir, parsed.output_dir, parsed.sort_by)


if __name__ == "__main__":
    main()
