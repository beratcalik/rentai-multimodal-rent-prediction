from __future__ import annotations

import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

import train_baseline as train_baseline_module
from train_baseline import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODEL_OUTPUT_PATH,
    PREPROCESSOR_OUTPUT_PATH,
    TARGET_COLUMN,
    calculate_metrics,
    configure_logging,
    load_dataset,
    parse_room_count,
    split_dataset,
)


LOGGER = logging.getLogger("baseline_error_analysis")

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT_DIR / "reports" / "baseline_error_analysis.md"
PLOTS_DIR = ROOT_DIR / "reports" / "plots"
ACTUAL_VS_PREDICTED_PLOT_PATH = PLOTS_DIR / "actual_vs_predicted.png"
RESIDUAL_HISTOGRAM_PLOT_PATH = PLOTS_DIR / "residual_histogram.png"
DISTRICT_ERROR_BAR_PLOT_PATH = PLOTS_DIR / "district_error_bar.png"

PRICE_BIN_EDGES = [0, 20_000, 30_000, 40_000, 50_000, 75_000, np.inf]
PRICE_BIN_LABELS = [
    "0-20k TRY",
    "20k-30k TRY",
    "30k-40k TRY",
    "40k-50k TRY",
    "50k-75k TRY",
    "75k+ TRY",
]
M2_BIN_EDGES = [0, 75, 100, 125, 150, 200, np.inf]
M2_BIN_LABELS = [
    "0-75 m2",
    "75-100 m2",
    "100-125 m2",
    "125-150 m2",
    "150-200 m2",
    "200+ m2",
]

STOPWORDS = {
    "acik",
    "ama",
    "ancak",
    "art",
    "ayni",
    "az",
    "baz",
    "beri",
    "bir",
    "biraz",
    "biz",
    "bu",
    "buna",
    "bunun",
    "cok",
    "da",
    "daha",
    "de",
    "degil",
    "den",
    "en",
    "gibi",
    "hem",
    "her",
    "icin",
    "ile",
    "ise",
    "kadar",
    "kar",
    "olan",
    "olarak",
    "olup",
    "sonra",
    "sadece",
    "seklinde",
    "siz",
    "su",
    "tum",
    "uzere",
    "ve",
    "veya",
    "ya",
    "yani",
}

KEYWORD_GROUPS = [
    ("esyali", ("esyali",)),
    ("luks", ("luks", "lux")),
    ("sifir", ("sifir",)),
    ("manzarali", ("manzarali",)),
    ("otopark", ("otopark",)),
    ("guvenlik", ("guvenlik",)),
    ("teras", ("teras",)),
    ("balkon", ("balkon",)),
    ("ankastre", ("ankastre",)),
    ("ebeveyn", ("ebeveyn",)),
    ("site", ("site",)),
    ("asansor", ("asansor",)),
    ("metro", ("metro",)),
]

BOILERPLATE_PHRASES = [
    "telefonu goster",
    "detayli bilgi",
    "arayiniz",
    "gayrimenkul",
    "kahve icmeye",
    "tapu",
    "kredi islemleri",
]


def ensure_output_directories() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts() -> tuple[Any, Any]:
    setattr(sys.modules["__main__"], "PropertyFeatureCleaner", train_baseline_module.PropertyFeatureCleaner)

    if not MODEL_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadi: {MODEL_OUTPUT_PATH}")
    if not PREPROCESSOR_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Preprocessor bulunamadi: {PREPROCESSOR_OUTPUT_PATH}")

    LOGGER.info("Model yukleniyor: %s", MODEL_OUTPUT_PATH)
    model = joblib.load(MODEL_OUTPUT_PATH)

    LOGGER.info("Preprocessor yukleniyor: %s", PREPROCESSOR_OUTPUT_PATH)
    preprocessor = joblib.load(PREPROCESSOR_OUTPUT_PATH)
    return model, preprocessor


def format_currency(value: float) -> str:
    return f"{value:,.0f}"


def collapse_whitespace(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def ascii_fold(text: Any) -> str:
    normalized = unicodedata.normalize("NFKD", collapse_whitespace(text))
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def tokenize(text: Any) -> list[str]:
    folded = ascii_fold(text)
    return re.findall(r"[a-z0-9]{2,}", folded)


def tokenize_without_stopwords(text: Any) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if token not in STOPWORDS and not token.isdigit()
    ]


def is_upper_heavy(text: Any, threshold: float = 0.70) -> bool:
    text_str = collapse_whitespace(text)
    letters = [char for char in text_str if char.isalpha()]
    if len(letters) < 20:
        return False
    uppercase_count = sum(1 for char in letters if char.isupper())
    return (uppercase_count / len(letters)) >= threshold


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_Veri yok_"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        escaped = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def series_to_markdown_table(
    series: pd.Series,
    value_header: str,
    digits: int = 2,
) -> str:
    rows: list[list[str]] = []
    for index, value in series.items():
        if isinstance(value, (float, np.floating)):
            rows.append([str(index), f"{float(value):,.{digits}f}"])
        else:
            rows.append([str(index), str(value)])
    return render_markdown_table(["Metric", value_header], rows)


def dataframe_to_markdown_table(dataframe: pd.DataFrame, digits: int = 2) -> str:
    if dataframe.empty:
        return "_Veri yok_"

    rows: list[list[str]] = []
    for _, row in dataframe.iterrows():
        rendered_row: list[str] = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                rendered_row.append(f"{float(value):,.{digits}f}")
            else:
                rendered_row.append(str(value))
        rows.append(rendered_row)
    return render_markdown_table(list(dataframe.columns), rows)


def build_prediction_frame(model: Any, preprocessor: Any) -> tuple[pd.DataFrame, dict[str, float]]:
    dataset = load_dataset(DATASET_PATH)
    splits = split_dataset(dataset)
    test_df = splits.test.copy()

    LOGGER.info("Test seti yeniden olusturuldu: %s ornek", len(test_df))
    transformed_test = preprocessor.transform(test_df[FEATURE_COLUMNS])
    predictions = model.predict(transformed_test)

    prediction_df = test_df.copy()
    prediction_df["actual_price_try"] = prediction_df[TARGET_COLUMN].astype(float)
    prediction_df["predicted_price_try"] = np.asarray(predictions, dtype=float)
    prediction_df["residual"] = (
        prediction_df["predicted_price_try"] - prediction_df["actual_price_try"]
    )
    prediction_df["abs_error"] = prediction_df["residual"].abs()
    prediction_df["ape_pct"] = (
        prediction_df["abs_error"] / prediction_df["actual_price_try"].clip(lower=1e-8)
    ) * 100.0
    prediction_df["rooms_numeric"] = prediction_df["rooms"].map(parse_room_count)
    prediction_df["rooms_group"] = prediction_df["rooms_numeric"].map(format_rooms_group)
    prediction_df["price_range"] = pd.cut(
        prediction_df["actual_price_try"],
        bins=PRICE_BIN_EDGES,
        labels=PRICE_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    prediction_df["m2_range"] = pd.cut(
        prediction_df["m2_gross"],
        bins=M2_BIN_EDGES,
        labels=M2_BIN_LABELS,
        right=False,
        include_lowest=True,
    )

    metrics = calculate_metrics(
        prediction_df["actual_price_try"],
        prediction_df["predicted_price_try"].to_numpy(dtype=float),
    )
    LOGGER.info(
        "Test tekrar dogrulama | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
        metrics["mape"],
    )
    return prediction_df, metrics


def format_rooms_group(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    numeric_value = int(value)
    if numeric_value >= 6:
        return "6+ rooms"
    return f"{numeric_value} rooms"


def summarize_group_errors(
    dataframe: pd.DataFrame,
    group_column: str,
    sort_by: str = "mae",
    ascending: bool = False,
) -> pd.DataFrame:
    summary_df = dataframe.copy()
    summary_df[group_column] = summary_df[group_column].astype("object").where(
        summary_df[group_column].notna(),
        "Unknown",
    )

    grouped = (
        summary_df.groupby(group_column, dropna=False)
        .agg(
            sample_count=("listing_id", "size"),
            mean_actual_price=("actual_price_try", "mean"),
            mae=("abs_error", "mean"),
            mape=("ape_pct", "mean"),
        )
        .reset_index()
        .rename(columns={group_column: "group"})
    )
    grouped = grouped.sort_values(
        by=[sort_by, "sample_count", "group"],
        ascending=[ascending, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return grouped


def summarize_binned_errors(
    dataframe: pd.DataFrame,
    group_column: str,
    label_order: Iterable[str],
) -> pd.DataFrame:
    grouped = summarize_group_errors(
        dataframe=dataframe,
        group_column=group_column,
        sort_by="sample_count",
        ascending=False,
    )
    grouped["group"] = pd.Categorical(grouped["group"], categories=list(label_order), ordered=True)
    grouped = grouped.sort_values("group").reset_index(drop=True)
    grouped["group"] = grouped["group"].astype(str)
    grouped = grouped[grouped["group"] != "nan"].reset_index(drop=True)
    return grouped


def select_worst_cases(dataframe: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    worst_df = dataframe.sort_values("abs_error", ascending=False).head(top_n).copy()
    worst_df["title_short"] = worst_df["title"].map(lambda text: collapse_whitespace(text)[:90])
    return worst_df[
        [
            "listing_id",
            "district",
            "neighborhood",
            "rooms",
            "m2_gross",
            "actual_price_try",
            "predicted_price_try",
            "residual",
            "abs_error",
            "ape_pct",
            "title_short",
        ]
    ].reset_index(drop=True)


def format_try_axis(axis: Any) -> None:
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))


def create_actual_vs_predicted_plot(dataframe: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    actual = dataframe["actual_price_try"].to_numpy(dtype=float)
    predicted = dataframe["predicted_price_try"].to_numpy(dtype=float)

    ax.scatter(actual, predicted, alpha=0.35, s=20, color="#1f77b4", edgecolors="none")
    lower_bound = min(actual.min(), predicted.min())
    upper_bound = max(actual.max(), predicted.max())
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle="--", color="#d62728", linewidth=1.5)
    ax.set_title("Baseline Test Set: Actual vs Predicted Rent")
    ax.set_xlabel("Actual price (TRY)")
    ax.set_ylabel("Predicted price (TRY)")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    fig.tight_layout()
    fig.savefig(ACTUAL_VS_PREDICTED_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_residual_histogram(dataframe: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    residuals = dataframe["residual"].to_numpy(dtype=float)
    ax.hist(residuals, bins=40, color="#4c78a8", alpha=0.85)
    ax.axvline(0, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title("Baseline Test Set: Residual Distribution")
    ax.set_xlabel("Residual (Predicted - Actual)")
    ax.set_ylabel("Listing count")
    ax.grid(alpha=0.2, axis="y")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    fig.tight_layout()
    fig.savefig(RESIDUAL_HISTOGRAM_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_district_error_bar_chart(district_summary: pd.DataFrame) -> None:
    plot_df = district_summary[district_summary["sample_count"] >= 5].head(12).copy()
    plot_df = plot_df.sort_values("mae", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["group"], plot_df["mae"], color="#f58518", alpha=0.9)
    ax.set_title("District-Level Test MAE")
    ax.set_xlabel("MAE (TRY)")
    ax.set_ylabel("District")
    ax.grid(alpha=0.2, axis="x")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))

    for index, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row["mae"],
            index,
            f"  n={int(row['sample_count'])}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(DISTRICT_ERROR_BAR_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_length_summary(text_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    normalized = text_series.fillna("").map(collapse_whitespace)
    char_lengths = normalized.str.len()
    word_lengths = normalized.str.split().str.len()

    summary = pd.Series(
        {
            "count": int(len(normalized)),
            "blank_count": int((normalized == "").sum()),
            "blank_ratio_pct": float((normalized == "").mean() * 100.0),
            "char_mean": float(char_lengths.mean()),
            "char_median": float(char_lengths.median()),
            "char_p90": float(char_lengths.quantile(0.90)),
            "char_p95": float(char_lengths.quantile(0.95)),
            "char_max": float(char_lengths.max()),
            "word_mean": float(word_lengths.mean()),
            "word_median": float(word_lengths.median()),
            "word_p90": float(word_lengths.quantile(0.90)),
            "word_p95": float(word_lengths.quantile(0.95)),
            "word_max": float(word_lengths.max()),
        }
    )
    return summary, char_lengths


def summarize_length_bins(char_lengths: pd.Series, bins: list[float], labels: list[str]) -> pd.DataFrame:
    bucketed = pd.cut(char_lengths, bins=bins, labels=labels, right=False, include_lowest=True)
    counts = bucketed.value_counts(sort=False).rename_axis("length_range").reset_index(name="sample_count")
    counts["share_pct"] = (counts["sample_count"] / max(len(char_lengths), 1)) * 100.0
    return counts


def count_terms(text_series: pd.Series, top_n: int = 20) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for text in text_series.fillna("").astype(str):
        counter.update(tokenize_without_stopwords(text))
    rows = [{"token": token, "count": count} for token, count in counter.most_common(top_n)]
    return pd.DataFrame(rows)


def build_keyword_price_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    combined_text = (dataframe["title"].fillna("") + " " + dataframe["description"].fillna("")).astype(str)
    token_sets = [set(tokenize(text)) for text in combined_text]
    overall_mean_price = float(dataframe[TARGET_COLUMN].mean())
    overall_median_price = float(dataframe[TARGET_COLUMN].median())

    rows: list[dict[str, Any]] = []
    for label, aliases in KEYWORD_GROUPS:
        mask = np.array([any(alias in token_set for alias in aliases) for token_set in token_sets], dtype=bool)
        count = int(mask.sum())
        if count == 0:
            continue
        subset = dataframe.loc[mask, TARGET_COLUMN].astype(float)
        rows.append(
            {
                "keyword": label,
                "listing_count": count,
                "share_pct": float((count / len(dataframe)) * 100.0),
                "mean_price_try": float(subset.mean()),
                "median_price_try": float(subset.median()),
                "mean_price_delta_try": float(subset.mean() - overall_mean_price),
                "median_price_delta_try": float(subset.median() - overall_median_price),
            }
        )

    keyword_df = pd.DataFrame(rows)
    if keyword_df.empty:
        return keyword_df
    return keyword_df.sort_values("mean_price_delta_try", ascending=False).reset_index(drop=True)


def build_text_cleaning_summary(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    title_series = dataframe["title"].fillna("").astype(str)
    description_series = dataframe["description"].fillna("").astype(str)
    combined_description_folded = description_series.map(ascii_fold)

    issue_rows = [
        {
            "signal": "blank_description",
            "count": int((description_series.map(collapse_whitespace) == "").sum()),
            "share_pct": float((description_series.map(collapse_whitespace) == "").mean() * 100.0),
        },
        {
            "signal": "title_upper_heavy",
            "count": int(title_series.map(is_upper_heavy).sum()),
            "share_pct": float(title_series.map(is_upper_heavy).mean() * 100.0),
        },
        {
            "signal": "description_upper_heavy",
            "count": int(description_series.map(is_upper_heavy).sum()),
            "share_pct": float(description_series.map(is_upper_heavy).mean() * 100.0),
        },
        {
            "signal": "html_like_description",
            "count": int(description_series.str.contains(r"<[^>]+>", regex=True).sum()),
            "share_pct": float(description_series.str.contains(r"<[^>]+>", regex=True).mean() * 100.0),
        },
        {
            "signal": "url_like_description",
            "count": int(description_series.str.contains(r"https?://|www\.", regex=True).sum()),
            "share_pct": float(description_series.str.contains(r"https?://|www\.", regex=True).mean() * 100.0),
        },
    ]

    for phrase in BOILERPLATE_PHRASES:
        contains_phrase = combined_description_folded.str.contains(re.escape(phrase), regex=True)
        issue_rows.append(
            {
                "signal": f"phrase:{phrase}",
                "count": int(contains_phrase.sum()),
                "share_pct": float(contains_phrase.mean() * 100.0),
            }
        )

    summary_df = pd.DataFrame(issue_rows)
    recommendations: list[str] = []

    blank_ratio = float(summary_df.loc[summary_df["signal"] == "blank_description", "share_pct"].iloc[0])
    title_upper_ratio = float(summary_df.loc[summary_df["signal"] == "title_upper_heavy", "share_pct"].iloc[0])
    description_upper_ratio = float(summary_df.loc[summary_df["signal"] == "description_upper_heavy", "share_pct"].iloc[0])
    html_ratio = float(summary_df.loc[summary_df["signal"] == "html_like_description", "share_pct"].iloc[0])
    url_ratio = float(summary_df.loc[summary_df["signal"] == "url_like_description", "share_pct"].iloc[0])
    phone_phrase_ratio = float(
        summary_df.loc[summary_df["signal"] == "phrase:telefonu goster", "share_pct"].iloc[0]
    )
    detail_phrase_ratio = float(
        summary_df.loc[summary_df["signal"] == "phrase:detayli bilgi", "share_pct"].iloc[0]
    )
    agency_phrase_ratio = float(
        summary_df.loc[summary_df["signal"] == "phrase:gayrimenkul", "share_pct"].iloc[0]
    )

    if blank_ratio > 0:
        recommendations.append(
            f"Description tarafinda bos kayit var ({blank_ratio:.2f}%). Text branch icin bos metinleri ayri ele almak gerekir."
        )
    if title_upper_ratio >= 50 or description_upper_ratio >= 50:
        recommendations.append(
            "Metinlerin buyuk bolumu buyuk harf agirlikli. Lowercase normalizasyonu ve unicode-folding faydali olur."
        )
    if phone_phrase_ratio >= 20 or detail_phrase_ratio >= 15 or agency_phrase_ratio >= 20:
        recommendations.append(
            "Iletisim ve emlak-ofisi boilerplate kaliplari yuksek. `telefonu goster`, `detayli bilgi`, ofis tanitimi gibi kaliplari temizlemek mantikli."
        )
    if html_ratio < 1 and url_ratio < 1:
        recommendations.append(
            "HTML ve URL temizligi dusuk oncelikli; ana kazanc boilerplate ve case normalization tarafinda."
        )
    else:
        recommendations.append(
            "Az sayida HTML/URL kalibi var; regex ile basit strip islemi eklenmeli."
        )

    return summary_df, recommendations


def build_report(
    prediction_df: pd.DataFrame,
    metrics: dict[str, float],
    district_summary: pd.DataFrame,
    price_summary: pd.DataFrame,
    m2_summary: pd.DataFrame,
    rooms_summary: pd.DataFrame,
    worst_cases: pd.DataFrame,
    title_length_summary: pd.Series,
    title_length_bins: pd.DataFrame,
    description_length_summary: pd.Series,
    description_length_bins: pd.DataFrame,
    title_top_words: pd.DataFrame,
    description_top_words: pd.DataFrame,
    keyword_price_table: pd.DataFrame,
    text_cleaning_summary: pd.DataFrame,
    cleaning_recommendations: list[str],
) -> str:
    title_stats_df = title_length_summary.rename("value").reset_index().rename(columns={"index": "metric"})
    description_stats_df = description_length_summary.rename("value").reset_index().rename(columns={"index": "metric"})

    plot_path_lines = [
        f"- Actual vs predicted scatter: `{ACTUAL_VS_PREDICTED_PLOT_PATH}`",
        f"- Residual histogram: `{RESIDUAL_HISTOGRAM_PLOT_PATH}`",
        f"- District MAE bar chart: `{DISTRICT_ERROR_BAR_PLOT_PATH}`",
    ]

    report_lines = [
        "# Baseline Error Analysis",
        "",
        "## Ozet",
        "",
        f"- Dataset: `{DATASET_PATH}`",
        f"- Yuklenen model: `{MODEL_OUTPUT_PATH}`",
        f"- Yuklenen preprocessor: `{PREPROCESSOR_OUTPUT_PATH}`",
        f"- Test ornek sayisi: **{len(prediction_df):,}**",
        "",
        render_markdown_table(
            ["Metric", "Value"],
            [
                ["MAE", f"{metrics['mae']:,.2f}"],
                ["RMSE", f"{metrics['rmse']:,.2f}"],
                ["R2", f"{metrics['r2']:,.4f}"],
                ["MAPE (%)", f"{metrics['mape']:,.2f}"],
            ],
        ),
        "",
        "## Test Set Error Breakdowns",
        "",
        "### District bazli MAE / MAPE",
        "",
        dataframe_to_markdown_table(district_summary.rename(columns={"group": "district"}), digits=2),
        "",
        "### Fiyat araligi bazli MAE / MAPE",
        "",
        dataframe_to_markdown_table(price_summary.rename(columns={"group": "price_range"}), digits=2),
        "",
        "### m2 araligi bazli MAE / MAPE",
        "",
        dataframe_to_markdown_table(m2_summary.rename(columns={"group": "m2_range"}), digits=2),
        "",
        "### Rooms bazli MAE / MAPE",
        "",
        dataframe_to_markdown_table(rooms_summary.rename(columns={"group": "rooms_group"}), digits=2),
        "",
        "### En yuksek 30 hata yapan ilan",
        "",
        dataframe_to_markdown_table(worst_cases, digits=2),
        "",
        "## Uretilen Plotlar",
        "",
        *plot_path_lines,
        "",
        "## Text Branch Hazirlik Analizi",
        "",
        "### Description uzunluk ozeti",
        "",
        dataframe_to_markdown_table(description_stats_df, digits=2),
        "",
        "### Description uzunluk dagilimi (character bins)",
        "",
        dataframe_to_markdown_table(description_length_bins, digits=2),
        "",
        "### Title uzunluk ozeti",
        "",
        dataframe_to_markdown_table(title_stats_df, digits=2),
        "",
        "### Title uzunluk dagilimi (character bins)",
        "",
        dataframe_to_markdown_table(title_length_bins, digits=2),
        "",
        "### Bos description var mi?",
        "",
        f"- Bos description sayisi: **{int(description_length_summary['blank_count'])}**",
        f"- Bos description orani: **{float(description_length_summary['blank_ratio_pct']):.2f}%**",
        "",
        "### En sik gecen title kelimeleri",
        "",
        dataframe_to_markdown_table(title_top_words, digits=2),
        "",
        "### En sik gecen description kelimeleri",
        "",
        dataframe_to_markdown_table(description_top_words, digits=2),
        "",
        "### Text temizleme gerekip gerekmedigi",
        "",
        dataframe_to_markdown_table(text_cleaning_summary, digits=2),
        "",
        *[f"- {recommendation}" for recommendation in cleaning_recommendations],
        "",
        "### Fiyatla iliskili olabilecek kelimeler icin basit analiz",
        "",
        dataframe_to_markdown_table(keyword_price_table, digits=2),
    ]
    return "\n".join(report_lines)


def main() -> int:
    configure_logging()
    ensure_output_directories()
    model, preprocessor = load_artifacts()

    prediction_df, metrics = build_prediction_frame(model=model, preprocessor=preprocessor)
    district_summary = summarize_group_errors(prediction_df, "district")
    price_summary = summarize_binned_errors(prediction_df, "price_range", PRICE_BIN_LABELS)
    m2_summary = summarize_binned_errors(prediction_df, "m2_range", M2_BIN_LABELS)
    rooms_summary = summarize_group_errors(prediction_df, "rooms_group")
    worst_cases = select_worst_cases(prediction_df, top_n=30)

    create_actual_vs_predicted_plot(prediction_df)
    create_residual_histogram(prediction_df)
    create_district_error_bar_chart(district_summary)

    full_dataset = load_dataset(DATASET_PATH)
    title_length_summary, title_char_lengths = build_length_summary(full_dataset["title"])
    description_length_summary, description_char_lengths = build_length_summary(full_dataset["description"])
    title_length_bins = summarize_length_bins(
        title_char_lengths,
        bins=[0, 40, 60, 80, 100, np.inf],
        labels=["0-39", "40-59", "60-79", "80-99", "100+"],
    )
    description_length_bins = summarize_length_bins(
        description_char_lengths,
        bins=[0, 100, 300, 600, 1000, 2000, np.inf],
        labels=["0-99", "100-299", "300-599", "600-999", "1000-1999", "2000+"],
    )
    title_top_words = count_terms(full_dataset["title"], top_n=20)
    description_top_words = count_terms(full_dataset["description"], top_n=20)
    keyword_price_table = build_keyword_price_table(full_dataset)
    text_cleaning_summary, cleaning_recommendations = build_text_cleaning_summary(full_dataset)

    report_body = build_report(
        prediction_df=prediction_df,
        metrics=metrics,
        district_summary=district_summary,
        price_summary=price_summary,
        m2_summary=m2_summary,
        rooms_summary=rooms_summary,
        worst_cases=worst_cases,
        title_length_summary=title_length_summary,
        title_length_bins=title_length_bins,
        description_length_summary=description_length_summary,
        description_length_bins=description_length_bins,
        title_top_words=title_top_words,
        description_top_words=description_top_words,
        keyword_price_table=keyword_price_table,
        text_cleaning_summary=text_cleaning_summary,
        cleaning_recommendations=cleaning_recommendations,
    )

    LOGGER.info("Rapor kaydediliyor: %s", REPORT_PATH)
    REPORT_PATH.write_text(report_body, encoding="utf-8")
    LOGGER.info("Tamamlandi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
