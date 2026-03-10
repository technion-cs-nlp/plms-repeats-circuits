from __future__ import annotations
import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plms_repeats_circuits.EAP.circuit_selection import (
    select_circuit_neurons,
    select_circuit_nodes,
)
from plms_repeats_circuits.EAP.graph import (
    AttentionNode,
    Graph,
    InputNode,
    LogitNode,
    MLPNode,
    MLPWithNeuronNode,
    NeuronGraph,
)
from plms_repeats_circuits.utils.counterfactuals_config import COUNTERFACTUAL_METHODS

REPEAT_TYPE_PATHS = {
    "identical": "identical",
    "approximate": "approximate",
    "synthetic": "synthetic",
}


@dataclass
class ComponentEntry:
    component_id: str
    metadata: Dict[str, object]
    per_circuit: Dict[int, Dict[str, object]]  # seed -> {in_graph, score, score_sign}


def _exp_prefix(repeat_type: str, counterfactual_type: str) -> str:
    return f"{repeat_type}_counterfactual_{counterfactual_type}"


def _output_dir(
    results_root: Path,
    repeat_type: str,
    model_type: str,
    graph_type: str,
    seed: int,
) -> Path:
    folder = REPEAT_TYPE_PATHS.get(repeat_type, repeat_type)
    return results_root / "circuit_discovery" / folder / model_type / graph_type / f"seed_{seed}"


def _find_json_for_counterfactual(
    output_dir: Path,
    exp_prefix: str,
    graph_type: str,
    seed: int,
) -> Optional[Tuple[str, Path]]:
    """Find the single JSON matching our counterfactual (exp_prefix). Returns (circuit_id, json_path) or None."""
    import re
    candidates = [
        p for p in output_dir.glob(f"{exp_prefix}_circuit*.json")
        if f"seed{seed}" in p.stem
        and ("_nodes.json" in p.name or "graph_type_nodes" in p.name if graph_type == "nodes"
             else "_neurons.json" in p.name or "graph_type_neurons" in p.name)
    ]
    result: List[Tuple[str, Path]] = []
    for p in candidates:
        match = re.search(r"_circuit([a-f0-9-]+)_", p.stem)
        if match:
            result.append((match.group(1), p))
    if len(result) == 0:
        return None
    if len(result) > 1:
        raise ValueError(
            f"Expected exactly one JSON for counterfactual {exp_prefix} seed={seed} in {output_dir}, found {len(result)}"
        )
    return result[0]


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "1", "yes", "y"}:
            return True
        if val in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _summarize_values(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return (np.nan, np.nan)
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _get_component_type(node) -> str:
    if isinstance(node, AttentionNode):
        return "attention"
    if isinstance(node, (MLPNode, MLPWithNeuronNode)):
        return "mlp"
    if isinstance(node, InputNode):
        return "input"
    if isinstance(node, LogitNode):
        return "logit"
    return node.__class__.__name__


def _score_sign(score: float) -> float:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return np.nan
    if abs(score) < 1e-12:
        return 0.0
    return 1.0 if score > 0 else -1.0


def process_nodes_mode(
    results_root: Path,
    repeat_type: str,
    model_type: str,
    counterfactual_type: str,
    seeds: List[int],
) -> Tuple[Dict[str, ComponentEntry], List[int]]:
    """Process nodes circuits. Returns (components, processed_seeds)."""
    components: Dict[str, ComponentEntry] = {}
    processed_seeds: List[int] = []
    exp_prefix = _exp_prefix(repeat_type, counterfactual_type)

    for seed in seeds:
        output_dir = _output_dir(results_root, repeat_type, model_type, "nodes", seed)
        match = _find_json_for_counterfactual(output_dir, exp_prefix, "nodes", seed)
        if match is None:
            print(f"  [nodes] repeat={repeat_type} seed={seed}: no JSON for counterfactual", flush=True)
            continue
        circuit_id, json_path = match

        info_path = output_dir / "circuit_info_nodes.csv"
        if not info_path.exists():
            print(f"  [nodes] repeat={repeat_type} seed={seed}: circuit_info missing -> {info_path}", flush=True)
            continue
        info_df = pd.read_csv(info_path)
        info_row = info_df[info_df["circuit_id"] == circuit_id]
        if info_row.empty:
            print(f"  [nodes] repeat={repeat_type} seed={seed}: circuit_id {circuit_id} not in circuit_info", flush=True)
            continue
        row = info_row.iloc[0]

        try:
            graph = Graph.from_json(str(json_path))
        except Exception as exc:
            print(f"  [nodes] repeat={repeat_type} seed={seed}: failed to load JSON: {exc}", flush=True)
            continue

        sel_method = str(row.get("circuit_selection_method", "top_n_abs")).strip()
        n_nodes_val = row.get("real_n_nodes", row.get("n_nodes", np.nan))
        if pd.isna(n_nodes_val):
            print(f"  [nodes] repeat={repeat_type} seed={seed}: no n_nodes", flush=True)
            continue
        n_nodes = int(float(n_nodes_val))

        if sel_method not in {"top_n", "top_n_abs"}:
            print(f"  [nodes] repeat={repeat_type} seed={seed}: unsupported selection {sel_method}", flush=True)
            continue
        select_circuit_nodes(graph, sel_method, n_nodes, log=None)

        scored_nodes = [
            n for n in graph.nodes.values()
            if n.score is not None and math.isfinite(float(n.score))
        ]
        if not scored_nodes:
            print(f"  [nodes] repeat={repeat_type} seed={seed}: no scored nodes", flush=True)
            continue

        print(f"  [nodes] repeat={repeat_type} seed={seed}: {len([n for n in scored_nodes if n.in_graph])} / {len(scored_nodes)} nodes in circuit", flush=True)
        processed_seeds.append(seed)

        for node in scored_nodes:
            raw_score = float(node.score)
            component_id = node.name
            metadata = {
                "component_type": _get_component_type(node),
                "layer": node.layer if hasattr(node, "layer") else None,
            }
            if isinstance(node, AttentionNode):
                metadata["head"] = node.head
            if component_id not in components:
                components[component_id] = ComponentEntry(
                    component_id=component_id,
                    metadata=metadata,
                    per_circuit={},
                )
            else:
                for k, v in metadata.items():
                    components[component_id].metadata.setdefault(k, v)
            components[component_id].per_circuit[seed] = {
                "in_graph": bool(node.in_graph),
                "score": raw_score,
                "score_sign": _score_sign(raw_score),
            }

    return components, processed_seeds


def process_neurons_mode(
    results_root: Path,
    repeat_type: str,
    model_type: str,
    counterfactual_type: str,
    seeds: List[int],
) -> Tuple[Dict[str, ComponentEntry], List[int]]:
    """Process neurons circuits. Returns (components, processed_seeds)."""
    components: Dict[str, ComponentEntry] = {}
    processed_seeds: List[int] = []
    exp_prefix = _exp_prefix(repeat_type, counterfactual_type)

    for seed in seeds:
        output_dir = _output_dir(results_root, repeat_type, model_type, "neurons", seed)
        match = _find_json_for_counterfactual(output_dir, exp_prefix, "neurons", seed)
        if match is None:
            print(f"  [neurons] repeat={repeat_type} seed={seed}: no JSON for counterfactual", flush=True)
            continue
        circuit_id, json_path = match

        info_path = output_dir / "circuit_info_neurons.csv"
        if not info_path.exists():
            print(f"  [neurons] repeat={repeat_type} seed={seed}: circuit_info missing -> {info_path}", flush=True)
            continue
        info_df = pd.read_csv(info_path)
        info_row = info_df[info_df["circuit_id"] == circuit_id]
        if info_row.empty:
            print(f"  [neurons] repeat={repeat_type} seed={seed}: circuit_id {circuit_id} not in circuit_info", flush=True)
            continue
        row = info_row.iloc[0]

        try:
            graph = NeuronGraph.from_json(str(json_path))
        except Exception as exc:
            print(f"  [neurons] repeat={repeat_type} seed={seed}: failed to load JSON: {exc}", flush=True)
            continue

        n_nodes_first = row.get("n_nodes_as_first_step_for_neurons", np.nan)
        n_neurons = row.get("n_neurons", np.nan)
        if pd.isna(n_nodes_first) or pd.isna(n_neurons):
            print(f"  [neurons] repeat={repeat_type} seed={seed}: missing n_nodes_first or n_neurons", flush=True)
            continue
        n_nodes_first = int(float(n_nodes_first))
        n_neurons = int(float(n_neurons))
        is_per_layer = _parse_bool(row.get("is_per_layer", False))
        sel_method = str(row.get("circuit_selection_method", "top_n_abs")).strip()

        select_circuit_neurons(
            graph,
            sel_method,
            n_neurons,
            n_nodes_first,
            is_per_layer,
            log=None,
        )

        full_layer_scores: Dict[int, np.ndarray] = {}
        selected_sets: Dict[int, Set[int]] = {}
        for node in graph.nodes.values():
            if isinstance(node, MLPWithNeuronNode):
                full_layer_scores[node.layer] = np.asarray(
                    node.neurons_scores, dtype=np.float32
                ) if node.neurons_scores is not None else np.array([], dtype=np.float32)
                if node.neurons_indicies_in_graph is not None:
                    selected_sets[node.layer] = set(int(i) for i in node.neurons_indicies_in_graph.tolist())
                else:
                    selected_sets[node.layer] = set()

        total_selected = sum(len(s) for s in selected_sets.values())
        print(f"  [neurons] repeat={repeat_type} seed={seed}: {total_selected} neurons in circuit", flush=True)
        processed_seeds.append(seed)

        for layer, scores in full_layer_scores.items():
            selected_layer = selected_sets.get(layer, set())
            for idx, raw_score_val in enumerate(scores):
                component_id = f"m{layer}_n{idx}"
                raw_score = float(raw_score_val)
                in_graph = idx in selected_layer
                if component_id not in components:
                    components[component_id] = ComponentEntry(
                        component_id=component_id,
                        metadata={
                            "component_type": "neuron",
                            "layer": layer,
                            "neuron_idx": idx,
                        },
                        per_circuit={},
                    )
                components[component_id].per_circuit[seed] = {
                    "in_graph": in_graph,
                    "score": raw_score,
                    "score_sign": _score_sign(raw_score),
                }

    return components, processed_seeds


def build_dataframe(
    components: Dict[str, ComponentEntry],
    repeat_type: str,
    processed_seeds: List[int],
    graph_type: str,
) -> pd.DataFrame:
    """Build output DataFrame with user-specified columns."""
    filtered = {
        cid: e for cid, e in components.items()
        if any(d.get("in_graph", False) for d in e.per_circuit.values())
    }
    records: List[Dict[str, object]] = []
    denom = len(processed_seeds)

    for component_id, entry in sorted(filtered.items()):
        row: Dict[str, object] = {
            "component_id": component_id,
            **{k: v for k, v in entry.metadata.items() if v is not None},
        }
        scores_for_repeat: List[float] = []
        for seed in processed_seeds:
            d = entry.per_circuit.get(seed, {})
            in_graph = d.get("in_graph", False)
            score = d.get("score", np.nan)
            sign = d.get("score_sign", np.nan)
            row[f"{repeat_type}_{seed}_in_graph"] = int(in_graph)
            row[f"{repeat_type}_{seed}_score"] = score if score is not None else np.nan
            row[f"{repeat_type}_{seed}_score_sign"] = sign if sign is not None else np.nan
            if in_graph and score is not None and not (isinstance(score, float) and np.isnan(score)):
                scores_for_repeat.append(float(score))

        in_graph_count = sum(1 for s in processed_seeds if entry.per_circuit.get(s, {}).get("in_graph", False))
        row[f"{repeat_type}_total_circuits"] = denom
        row[f"{repeat_type}_in_graph_count"] = in_graph_count
        row[f"{repeat_type}_ratio_in_graph"] = in_graph_count / denom if denom else np.nan
        mean_s, std_s = _summarize_values(scores_for_repeat)
        row[f"{repeat_type}_mean_score"] = mean_s
        row[f"{repeat_type}_std_score"] = std_s
        if mean_s is not None and not np.isnan(mean_s) and abs(mean_s) >= 1e-12:
            row[f"{repeat_type}_cv_score"] = std_s / abs(mean_s) if std_s is not None else np.nan
        else:
            row[f"{repeat_type}_cv_score"] = np.nan
        records.append(row)

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    valid_methods = [m["name"] for m in COUNTERFACTUAL_METHODS]
    parser = argparse.ArgumentParser(
        description="Summarize component recurrence across seeds for nodes or neurons circuits."
    )
    parser.add_argument(
        "--graph_type",
        choices=["nodes", "neurons"],
        required=True,
        help="Circuit granularity: nodes or neurons.",
    )
    parser.add_argument(
        "--repeat_type",
        required=True,
        choices=["identical", "approximate", "synthetic"],
        help="Repeat type (one per run).",
    )
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["esm3", "esm-c"],
        help="Model type.",
    )
    parser.add_argument(
        "--counterfactual_type",
        required=True,
        choices=valid_methods,
        help=f"Counterfactual type. One of: {valid_methods}",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds to include.",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root for circuit_discovery outputs and component_recurrence results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(args.seeds)

    print("=" * 80, flush=True)
    print("Component Recurrence Stats — start", flush=True)
    print("=" * 80, flush=True)
    print(f"graph_type: {args.graph_type}", flush=True)
    print(f"repeat_type: {args.repeat_type}", flush=True)
    print(f"model_type: {args.model_type}", flush=True)
    print(f"counterfactual_type: {args.counterfactual_type}", flush=True)
    print(f"seeds: {seeds}", flush=True)
    print("=" * 80, flush=True)

    if args.graph_type == "nodes":
        components, processed_seeds = process_nodes_mode(
            args.results_root,
            args.repeat_type,
            args.model_type,
            args.counterfactual_type,
            seeds,
        )
    else:
        components, processed_seeds = process_neurons_mode(
            args.results_root,
            args.repeat_type,
            args.model_type,
            args.counterfactual_type,
            seeds,
        )

    if not processed_seeds:
        print("No circuits processed; nothing to save.", flush=True)
        sys.exit(1)

    df = build_dataframe(components, args.repeat_type, processed_seeds, args.graph_type)
    sort_cols = ["component_type", "layer", "component_id"]
    if "neuron_idx" in df.columns:
        sort_cols.insert(2, "neuron_idx")
    df.sort_values([c for c in sort_cols if c in df.columns], inplace=True, na_position="last")

    output_dir = (
        args.results_root / "component_recurrence" / args.model_type / args.counterfactual_type
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.graph_type}_recurrence_{args.repeat_type}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}", flush=True)
    print("=" * 80, flush=True)
    print("Component Recurrence Stats — completed", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
