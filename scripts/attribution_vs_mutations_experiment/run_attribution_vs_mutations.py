"""Run node EAP-IG attribution vs number of BLOSUM mutations in the non-masked repeat."""

from __future__ import annotations

import logging
import os
from typing import Dict

import pandas as pd

from circuit_discovery_imports import EAPDataset
from plms_repeats_circuits.EAP.attribute import attribute
from plms_repeats_circuits.EAP.graph import Graph, GraphType
from plms_repeats_circuits.utils.esm_utils import load_model, load_tokenizer_by_model_type
from plms_repeats_circuits.utils.experiment_utils import set_random_seed
from plms_repeats_circuits.utils.model_utils import get_device
from plms_repeats_circuits.utils.patching_metrics import create_loss_and_metric

from utils import create_mutated_induction_dataset, load_relevant_components


def _extract_node_scores(graph: Graph, important_nodes: pd.DataFrame) -> Dict[str, float]:
    node_scores: Dict[str, float] = {}
    for _, comp_row in important_nodes.iterrows():
        component_id = comp_row["component_id"]
        if component_id not in graph.nodes:
            logging.warning("Component %s not found in graph", component_id)
            continue
        node = graph.nodes[component_id]
        if node.score is not None:
            node_scores[component_id] = float(node.score)
        else:
            logging.warning("Node %s has no score", component_id)
    return node_scores


def _run_attribution_for_mutations(
    model,
    tokenizer,
    df: pd.DataFrame,
    n_mutations: int,
    device,
    metric_loss,
    eap_ig_steps: int,
    batch_size: int,
    aggregation: str,
    important_nodes: pd.DataFrame,
    metric_name: str,
) -> Dict[str, float]:
    mutated_df = create_mutated_induction_dataset(
        original_df=df,
        n_mutations=n_mutations,
        tokenizer=tokenizer,
        metric=metric_name,
    )
    if len(mutated_df) == 0:
        logging.warning("No valid samples after mutation for n_mutations=%s", n_mutations)
        return {}

    dataset = EAPDataset(mutated_df)
    dataloader = dataset.to_dataloader(batch_size=batch_size)
    graph = Graph.from_model(model, graph_type=GraphType.Nodes)

    attribute(
        model=model,
        graph=graph,
        dataloader=dataloader,
        metric=metric_loss,
        device=device,
        aggregation=aggregation,
        method="EAP-IG",
        quiet=False,
        abs_per_pos=False,
        are_clean_logits_needed=False,
        eap_ig_steps=eap_ig_steps,
    )
    return _extract_node_scores(graph, important_nodes)


def run_attribution_vs_mutations(
    dataset_csv: str,
    component_recurrence_csv: str,
    out_csv_path: str,
    model_type: str,
    component_recurrence_repeat_type: str,
    repeat_length: int,
    n_samples: int = 40,
    random_state: int = 42,
    component_recurrence_threshold: float = 0.8,
    eap_ig_steps: int = 5,
    batch_size: int = 1,
    aggregation: str = "sum",
    metric: str = "log_prob",
) -> None:
    """Compute node attribution scores for n_mutations = 0 .. repeat_length - 1."""
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    log_path = os.path.join(os.path.dirname(out_csv_path) or ".", f"repeat_len_{repeat_length}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
        force=True,
    )

    set_random_seed(random_state)
    device = get_device()
    model = load_model(
        model_type=model_type,
        device=device,
        use_transformer_lens_model=True,
        cache_attention_activations=True,
        cache_mlp_activations=True,
        output_type="sequence",
        cache_attn_pattern=False,
        split_qkv_input=True,
    )
    tokenizer = load_tokenizer_by_model_type(model_type=model_type, model=model)

    df = pd.read_csv(dataset_csv)
    df["repeat_length"] = df["repeat_length"].astype(int)
    df = df[df["repeat_length"] == repeat_length]
    logging.info("Filtered to %d samples with repeat_length=%s", len(df), repeat_length)

    if len(df) < n_samples:
        logging.warning("Only %d samples available, using all", len(df))
        n_samples = len(df)
    df = df.sample(n=n_samples, random_state=random_state)

    important_nodes = load_relevant_components(
        component_recurrence_csv,
        component_recurrence_repeat_type,
        component_recurrence_threshold,
    )
    loss, _metric_fn = create_loss_and_metric(metric)

    max_mutations = repeat_length - 1
    node_results: list[dict] = []

    for n_mutations in range(0, max_mutations + 1):
        logging.info("Processing n_mutations = %s", n_mutations)
        try:
            node_scores = _run_attribution_for_mutations(
                model=model,
                tokenizer=tokenizer,
                df=df,
                n_mutations=n_mutations,
                device=device,
                metric_loss=loss,
                eap_ig_steps=eap_ig_steps,
                batch_size=batch_size,
                aggregation=aggregation,
                important_nodes=important_nodes,
                metric_name=metric,
            )
            for component_id, score in node_scores.items():
                comp_info = important_nodes[
                    important_nodes["component_id"] == component_id
                ]
                row0 = comp_info.iloc[0] if len(comp_info) > 0 else None
                node_results.append(
                    {
                        "n_mutations": n_mutations,
                        "component_id": component_id,
                        "component_type": (
                            row0["component_type"] if row0 is not None else "unknown"
                        ),
                        "layer": row0["layer"] if row0 is not None else None,
                        "head": row0.get("head") if row0 is not None else None,
                        "score": score,
                        "repeat_length": repeat_length,
                        "n_samples": len(df),
                    }
                )
        except Exception:
            logging.exception("Error processing n_mutations=%s", n_mutations)

    pd.DataFrame(node_results).to_csv(out_csv_path, index=False)
    logging.info("Saved %d node results to %s", len(node_results), out_csv_path)
