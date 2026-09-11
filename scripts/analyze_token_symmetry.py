#!/usr/bin/env python3
"""CPU-only token-symmetry analysis joined to coupled FedEx transfer results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

from fedllm_heterogeneity.data import read_examples
from fedllm_heterogeneity.diagnostics import (
    shared_token_mass,
    token_distribution,
    token_js_divergence,
    token_label_mutual_information,
    ubiquitous_token_mass,
    vocabulary_jaccard,
    weighted_token_overlap,
)
from fedllm_heterogeneity.partitioning import example_pool_hash
from fedllm_heterogeneity.types import CanonicalExample, SymmetryProfile


DOMAINS = ("general", "finance", "medical", "code")
TASKS = ("continuation", "span_reconstruction")
CELLS = tuple(f"{domain}/{task}" for domain in DOMAINS for task in TASKS)
SCOPES = ("context", "target", "full_prompt", "full_example")
SEEDS = (42, 43, 44)
EXPECTED_EXAMPLE_SHA256 = "fcc8ec5c979899b438406b7af5b0c7bbb6f386a34d63ea417183399cc1c961b0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_task_template(example: CanonicalExample) -> str:
    if example.task == "continuation":
        prefix = "Continue the following text.\n\n"
        suffix = "\n\nContinuation:"
    elif example.task == "span_reconstruction":
        prefix = "Reconstruct the missing text exactly.\n\n"
        suffix = "\n\nMissing text:"
    else:
        return example.prompt
    if not example.prompt.startswith(prefix) or not example.prompt.endswith(suffix):
        raise ValueError(f"Unexpected prompt template for {example.example_id}")
    return example.prompt[len(prefix) : -len(suffix)]


def scoped_text(example: CanonicalExample, scope: str) -> str:
    if scope == "context":
        return strip_task_template(example)
    if scope == "target":
        return example.target
    if scope == "full_prompt":
        return example.prompt
    if scope == "full_example":
        return example.prompt + "\n" + example.target
    raise ValueError(scope)


def cell(example: CanonicalExample) -> str:
    return f"{example.domain}/{example.task}"


def encoded_distributions(examples, tokenizer, scope):
    grouped_sequences: dict[str, list[list[int]]] = defaultdict(list)
    grouped_texts: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        value = scoped_text(example, scope)
        grouped_texts[cell(example)].append(value)
        grouped_sequences[cell(example)].append(
            tokenizer.encode(value, add_special_tokens=False)
        )
    distributions = {
        group: token_distribution(grouped_sequences[group]) for group in CELLS
    }
    return distributions, grouped_sequences, grouped_texts


def pair_row(left, right, scope, distributions):
    p = distributions[left]
    q = distributions[right]
    left_domain, left_task = left.split("/")
    right_domain, right_task = right.split("/")
    return {
        "left_cell": left,
        "right_cell": right,
        "scope": scope,
        "same_domain": left_domain == right_domain,
        "same_task": left_task == right_task,
        "support_shared_mass": shared_token_mass(p, q),
        "weighted_token_overlap": weighted_token_overlap(p, q),
        "vocabulary_jaccard": vocabulary_jaccard(p, q),
        "js_divergence_bits": token_js_divergence(p, q),
    }


def top_information_tokens(distributions, contributions, tokenizer, limit=30):
    rows = []
    for token_id, contribution in sorted(
        contributions.items(), key=lambda item: item[1], reverse=True
    )[:limit]:
        per_domain = {key: value[token_id] for key, value in distributions.items()}
        dominant = max(per_domain, key=per_domain.get)
        rows.append(
            {
                "token_id": token_id,
                "token": tokenizer.decode([token_id], skip_special_tokens=False),
                "mi_contribution_bits": contribution,
                "dominant_domain": dominant,
                "counts": per_domain,
            }
        )
    return rows


def structural_rows(examples):
    patterns = {
        "punctuation": re.compile(r"[^\w\s]", re.UNICODE),
        "digit": re.compile(r"\d"),
        "bracket": re.compile(r"[(){}\[\]]"),
        "operator": re.compile(r"[=+*/<>!-]"),
        "newline": re.compile(r"\n"),
    }
    rows = []
    for group in CELLS:
        subset = [item for item in examples if cell(item) == group]
        for scope in ("context", "target"):
            texts = [scoped_text(item, scope) for item in subset]
            chars = max(sum(len(text) for text in texts), 1)
            words = [len(text.split()) for text in texts]
            row = {
                "cell": group,
                "scope": scope,
                "examples": len(texts),
                "mean_whitespace_units": float(np.mean(words)),
                "median_whitespace_units": float(np.median(words)),
            }
            for name, pattern in patterns.items():
                row[f"{name}_per_100_chars"] = 100 * sum(
                    len(pattern.findall(text)) for text in texts
                ) / chars
            rows.append(row)
    return rows


def boilerplate_audit_rows(examples):
    """Measure known native-task instructions that survived controlled sampling."""

    patterns = {
        "finance_native_instruction_fragment": re.compile(
            r"(?i)(what is the sentiment of this news|please choose an answer from|choose an answer|answer from|negative/neutral|neutral/positive|strong negative|strong positive)"
        ),
        "finance_label_vocabulary": re.compile(
            r"(?i)\b(positive|negative|neutral)\b"
        ),
        "medical_native_instruction": re.compile(
            r"(?i)answer this question truthfully"
        ),
    }
    rows = []
    for group in CELLS:
        subset = [item for item in examples if cell(item) == group]
        for text_scope in ("prompt", "target"):
            for name, pattern in patterns.items():
                matched = sum(
                    bool(pattern.search(getattr(item, text_scope))) for item in subset
                )
                rows.append(
                    {
                        "cell": group,
                        "text_scope": text_scope,
                        "native_boilerplate": name,
                        "examples": len(subset),
                        "matched_examples": matched,
                        "matched_fraction": matched / len(subset),
                    }
                )
    return rows


def coupled_effect_rows(results: Path):
    rows = []
    for seed in SEEDS:
        directory = results / f"seed-{seed}__coupled__fedex_lora"
        artifact = directory / "artifacts" / "runs" / f"seed-{seed}" / "coupled" / "fedex_lora"
        summary = json.loads((artifact / "summary.json").read_text())
        final_round = json.loads((artifact / "rounds.json").read_text())[-1]
        transfer = final_round["functional_transfer"]
        loo = final_round["leave_one_out_aggregation"]
        if transfer["client_ids"] != loo["client_ids"] or transfer["cells"] != loo["cells"]:
            raise ValueError(f"Transfer/LOO alignment mismatch for seed {seed}")
        weights = summary["client_cell_weights"]
        for client_index, client_id in enumerate(transfer["client_ids"]):
            train_cells = [name for name, weight in weights[client_id].items() if weight > 0]
            if len(train_cells) != 1:
                raise ValueError(f"Coupled client {client_id} has {train_cells}")
            training_cell = train_cells[0]
            for eval_index, eval_cell in enumerate(transfer["cells"]):
                loss_delta = transfer["loss_delta_matrix"][client_index][eval_index]
                loo_delta = loo["loss_delta_without_client_minus_full_matrix"][client_index][eval_index]
                rows.append(
                    {
                        "seed": seed,
                        "client_id": client_id,
                        "training_cell": training_cell,
                        "eval_cell": eval_cell,
                        "direct_loss_delta": loss_delta,
                        "direct_benefit": -loss_delta,
                        "loo_loss_delta_without_minus_full": loo_delta,
                        "loo_benefit": loo_delta,
                    }
                )
    return pd.DataFrame(rows)


def correlation_record(frame, x, y):
    if len(frame) < 3 or frame[x].nunique() < 2 or frame[y].nunique() < 2:
        return {"n": len(frame), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pearson = pearsonr(frame[x], frame[y])
    spearman = spearmanr(frame[x], frame[y])
    return {
        "n": len(frame),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def cluster_regression(frame, outcome):
    work = frame.copy()
    z = (work.weighted_token_overlap - work.weighted_token_overlap.mean()) / work.weighted_token_overlap.std(ddof=0)
    x = np.column_stack(
        [
            np.ones(len(work)),
            z,
            work.same_domain.astype(float),
            work.same_task.astype(float),
            (work.seed == 43).astype(float),
            (work.seed == 44).astype(float),
        ]
    )
    y = work[outcome].to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    inverse = np.linalg.pinv(x.T @ x)
    residual = y - x @ beta
    cluster_ids = (work.training_cell + "->" + work.eval_cell).to_numpy()
    unique_clusters = sorted(set(cluster_ids))
    meat = np.zeros((x.shape[1], x.shape[1]))
    for cluster_id in unique_clusters:
        selected = cluster_ids == cluster_id
        score = x[selected].T @ residual[selected]
        meat += np.outer(score, score)
    correction = (len(unique_clusters) / (len(unique_clusters) - 1)) * (
        (len(work) - 1) / (len(work) - x.shape[1])
    )
    covariance = correction * inverse @ meat @ inverse
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0, None))
    names = ("intercept", "overlap_per_sd", "same_domain", "same_task", "seed_43", "seed_44")
    return {
        name: {"coefficient": float(coef), "cluster_robust_se": float(se)}
        for name, coef, se in zip(names, beta, standard_error)
    } | {
        "n": len(work),
        "clusters": len(unique_clusters),
        "cluster_unit": "ordered training-cell -> evaluation-cell pair",
        "overlap_sd": float(work.weighted_token_overlap.std(ddof=0)),
    }


def render_figure(pair_metrics, joined, destination):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.2))
    labels = [name.replace("/span_reconstruction", "/recon").replace("/continuation", "/cont") for name in CELLS]
    for ax, scope, title in zip(
        axes[0],
        ("context", "target"),
        ("A. Context overlap (outer task template removed*)", "B. Target-token overlap"),
    ):
        subset = pair_metrics[pair_metrics.scope == scope]
        matrix = pd.DataFrame(np.eye(len(CELLS)), index=CELLS, columns=CELLS)
        for row in subset.itertuples():
            matrix.loc[row.left_cell, row.right_cell] = row.weighted_token_overlap
            matrix.loc[row.right_cell, row.left_cell] = row.weighted_token_overlap
        sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", square=True,
                    linewidths=.45, cbar_kws={"label": "Frequency-weighted overlap"}, ax=ax)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, rotation=0, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, fontsize=13, weight="bold")

    plot = joined[(joined.scope == "context") & (~joined.self_pair)].groupby(
        ["training_cell", "eval_cell", "same_task"], as_index=False
    ).agg(weighted_token_overlap=("weighted_token_overlap", "first"),
          direct_benefit=("direct_benefit", "mean"), loo_benefit=("loo_benefit", "mean"))
    palette = {True: "#397A70", False: "#D88C46"}
    for ax, outcome, title in zip(
        axes[1],
        ("direct_benefit", "loo_benefit"),
        ("C. One-client update benefit", "D. Benefit retained in full aggregate (LOO)"),
    ):
        for same_task, group in plot.groupby("same_task"):
            ax.scatter(group.weighted_token_overlap, group[outcome], s=46,
                       alpha=.82, color=palette[same_task], edgecolor="white", linewidth=.5,
                       label="Same task" if same_task else "Different task")
            if len(group) >= 3 and group.weighted_token_overlap.nunique() > 1:
                coefficient = np.polyfit(group.weighted_token_overlap, group[outcome], 1)
                xs = np.linspace(group.weighted_token_overlap.min(), group.weighted_token_overlap.max(), 80)
                ax.plot(xs, np.polyval(coefficient, xs), color=palette[same_task], linewidth=1.5)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xlabel("Context-token overlap (0–1)")
        ax.set_ylabel("NLL improvement (positive helps)")
        ax.set_title(title, fontsize=13, weight="bold")
        ax.legend(frameon=False, fontsize=10)
    fig.suptitle("Token overlap is structured, but task identity explains transfer more clearly", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.005,
        "* Native finance/medical dataset instructions remain in the baseline context; this was discovered by the CPU audit.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, default=Path("tmp/controlled_examples.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("fedllm-results"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/symmetry"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    if sha256_file(args.examples) != EXPECTED_EXAMPLE_SHA256:
        raise ValueError("Canonical example SHA-256 does not match the frozen 12-run baseline")
    examples = [item for item in read_examples(args.examples) if item.split == "train"]
    pool_digest = example_pool_hash(examples)
    baseline = json.loads((ROOT / "baseline" / "fedex-qwen-3seed.json").read_text())
    if pool_digest != baseline["training_example_pool_hash"]:
        raise ValueError("Training example-pool hash does not match frozen baseline")
    if len(examples) != 4096 or Counter(cell(item) for item in examples) != Counter({name: 512 for name in CELLS}):
        raise ValueError("Expected exactly 512 training examples in each of eight cells")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer_path = snapshot_download(
        args.tokenizer, cache_dir=args.cache_dir, local_files_only=True
    )
    # Passing the resolved snapshot path prevents recent Transformers versions
    # from making an unnecessary model-card network request during tokenization.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, fix_mistral_regex=False
    )
    args.output.mkdir(parents=True, exist_ok=True)
    all_pair_rows = []
    cell_rows = []
    profiles = []
    information_reports = []
    distributions_by_scope = {}

    for scope in SCOPES:
        distributions, sequences, texts = encoded_distributions(examples, tokenizer, scope)
        distributions_by_scope[scope] = distributions
        pairwise = []
        for index, left in enumerate(CELLS):
            for right in CELLS[index + 1 :]:
                row = pair_row(left, right, scope, distributions)
                pairwise.append(row)
                all_pair_rows.append(row)
        counts = {name: len(sequences[name]) for name in CELLS}
        token_counts = {name: sum(distributions[name].values()) for name in CELLS}
        vocabulary_sizes = {name: len(distributions[name]) for name in CELLS}
        fertility = {
            name: token_counts[name] / max(sum(len(value.split()) for value in texts[name]), 1)
            for name in CELLS
        }
        for name in CELLS:
            cell_rows.append(
                {
                    "cell": name,
                    "scope": scope,
                    "examples": counts[name],
                    "tokens": token_counts[name],
                    "vocabulary_size": vocabulary_sizes[name],
                    "tokens_per_whitespace_unit": fertility[name],
                }
            )

        domain_information = {}
        for task in (*TASKS, "all_tasks"):
            selected = [name for name in CELLS if task == "all_tasks" or name.endswith("/" + task)]
            collapsed = {
                domain: sum((distributions[name] for name in selected if name.startswith(domain + "/")), Counter())
                for domain in DOMAINS
            }
            mi, contributions = token_label_mutual_information(collapsed)
            report = {
                "scope": scope,
                "task": task,
                "domain_mutual_information_bits": mi,
                "normalized_domain_information": mi / np.log2(len(DOMAINS)),
                "ubiquitous_token_mass": ubiquitous_token_mass(collapsed),
                "top_domain_informative_tokens": top_information_tokens(collapsed, contributions, tokenizer),
            }
            information_reports.append(report)
            domain_information[task] = {
                key: value for key, value in report.items() if key not in {"scope", "task"}
            }
        profile = SymmetryProfile(
            split="train",
            scope=scope,
            tokenizer=args.tokenizer,
            groups=CELLS,
            example_counts=counts,
            token_counts=token_counts,
            vocabulary_sizes=vocabulary_sizes,
            fertility_tokens_per_whitespace_unit=fertility,
            pairwise=tuple(pairwise),
            domain_information=domain_information,
        )
        profiles.append(asdict(profile))

    pair_metrics = pd.DataFrame(all_pair_rows)
    pair_metrics.to_csv(args.output / "cell_pair_metrics.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(args.output / "cell_profiles.csv", index=False)
    pd.DataFrame(structural_rows(examples)).to_csv(args.output / "structural_profiles.csv", index=False)
    boilerplate = pd.DataFrame(boilerplate_audit_rows(examples))
    boilerplate.to_csv(args.output / "boilerplate_audit.csv", index=False)
    (args.output / "token_profiles.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "canonical_examples_sha256": EXPECTED_EXAMPLE_SHA256,
                "training_example_pool_hash": pool_digest,
                "split": "train",
                "profiles": profiles,
                "domain_information_reports": information_reports,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )

    effect_rows = coupled_effect_rows(args.results)
    similarity_rows = []
    for scope, distributions in distributions_by_scope.items():
        for left in CELLS:
            for right in CELLS:
                similarity_rows.append(pair_row(left, right, scope, distributions))
    directed_similarity = pd.DataFrame(similarity_rows).rename(
        columns={"left_cell": "training_cell", "right_cell": "eval_cell"}
    )
    joined = effect_rows.merge(directed_similarity, on=["training_cell", "eval_cell"], validate="many_to_many")
    joined["self_pair"] = joined.training_cell == joined.eval_cell
    joined.to_csv(args.output / "coupled_transfer_with_symmetry.csv", index=False)

    pair_seed = joined.groupby(
        ["seed", "training_cell", "eval_cell", "scope", "same_domain", "same_task", "self_pair"],
        as_index=False,
    ).agg(
        weighted_token_overlap=("weighted_token_overlap", "first"),
        js_divergence_bits=("js_divergence_bits", "first"),
        vocabulary_jaccard=("vocabulary_jaccard", "first"),
        direct_benefit=("direct_benefit", "mean"),
        loo_benefit=("loo_benefit", "mean"),
    )
    pair_seed.to_csv(args.output / "cell_pair_transfer_summary.csv", index=False)

    correlations = {}
    regressions = {}
    for scope in SCOPES:
        off_diagonal = pair_seed[(pair_seed.scope == scope) & (~pair_seed.self_pair)]
        pair_average = off_diagonal.groupby(
            ["training_cell", "eval_cell", "same_domain", "same_task"], as_index=False
        ).agg(
            weighted_token_overlap=("weighted_token_overlap", "first"),
            direct_benefit=("direct_benefit", "mean"),
            loo_benefit=("loo_benefit", "mean"),
        )
        same_task_average = pair_average[(~pair_average.same_domain) & pair_average.same_task].copy()
        same_task_average["task"] = same_task_average.training_cell.str.split("/").str[1]
        same_task_average["domain_pair"] = same_task_average.apply(
            lambda row: ":".join(
                sorted([row.training_cell.split("/")[0], row.eval_cell.split("/")[0]])
            ),
            axis=1,
        )
        bidirectional = same_task_average.groupby(
            ["task", "domain_pair"], as_index=False
        ).agg(
            weighted_token_overlap=("weighted_token_overlap", "first"),
            direct_benefit=("direct_benefit", "mean"),
            loo_benefit=("loo_benefit", "mean"),
        )
        correlations[scope] = {
            "all_off_diagonal_direct": correlation_record(pair_average, "weighted_token_overlap", "direct_benefit"),
            "all_off_diagonal_loo": correlation_record(pair_average, "weighted_token_overlap", "loo_benefit"),
            "cross_domain_same_task_direct": correlation_record(same_task_average, "weighted_token_overlap", "direct_benefit"),
            "cross_domain_same_task_loo": correlation_record(same_task_average, "weighted_token_overlap", "loo_benefit"),
            "bidirectional_domain_pair_loo": correlation_record(bidirectional, "weighted_token_overlap", "loo_benefit"),
            "bidirectional_domain_pair_loo_by_task": {
                task: correlation_record(
                    bidirectional[bidirectional.task == task],
                    "weighted_token_overlap",
                    "loo_benefit",
                )
                for task in TASKS
            },
        }
        regressions[scope] = {
            "direct_benefit": cluster_regression(off_diagonal, "direct_benefit"),
            "loo_benefit": cluster_regression(off_diagonal, "loo_benefit"),
        }
        if scope == "context":
            bidirectional.to_csv(args.output / "same_task_cross_domain_pairs.csv", index=False)

    relationship = pair_seed[pair_seed.scope == "context"].groupby(
        ["same_domain", "same_task", "self_pair"], as_index=False
    ).agg(
        pairs=("direct_benefit", "size"),
        mean_overlap=("weighted_token_overlap", "mean"),
        mean_direct_benefit=("direct_benefit", "mean"),
        mean_loo_benefit=("loo_benefit", "mean"),
    )
    relationship.to_csv(args.output / "relationship_summary.csv", index=False)

    summary = {
        "integrity": {
            "canonical_examples_sha256": EXPECTED_EXAMPLE_SHA256,
            "training_example_pool_hash": pool_digest,
            "training_examples": len(examples),
            "examples_per_cell": 512,
            "coupled_client_cell_observations": len(effect_rows),
            "known_native_boilerplate_matches": boilerplate[
                boilerplate.matched_examples > 0
            ].to_dict(orient="records"),
        },
        "correlations": correlations,
        "regressions_cluster_robust_descriptive": regressions,
        "relationship_summary_context": relationship.to_dict(orient="records"),
        "interpretation_guardrail": "Observational association on one fixed example pool; not causal evidence for token symmetry.",
    }
    (args.output / "symmetry_transfer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    figure = args.results / "analysis" / "fig6-token-symmetry-transfer.png"
    render_figure(pair_metrics, joined, figure)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote CPU-only symmetry artifacts to {args.output} and {figure}")


if __name__ == "__main__":
    main()
