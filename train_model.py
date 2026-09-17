"""Entrena, valida y documenta el caso de previsión de tesorería."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cashflow_forecasting.modeling import (  # noqa: E402
    CashFlowForecaster,
    add_balance,
    evaluate_forecast,
    seasonal_naive_forecast,
)
from generador_datos import export_cash_flow_data, generate_cash_flow_data  # noqa: E402


DATA_PATH = Path("data/tesoreria_sintetica.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = Path("artifacts")
HORIZON = 60
VALIDATION_STARTS = ("2025-03-01", "2025-06-01", "2025-09-01")
TEST_START = pd.Timestamp("2025-12-31")


def load_data() -> pd.DataFrame:
    data = generate_cash_flow_data()
    export_cash_flow_data(data, numeric_path=DATA_PATH)
    data["Fecha"] = pd.to_datetime(data["Fecha"])
    return data


def forecast_method(method: str, train: pd.DataFrame, dates: pd.DatetimeIndex):
    if method == "gradient_boosting":
        model = CashFlowForecaster().fit(train)
        return add_balance(model.predict(dates), float(train["Saldo_Bancario"].iloc[-1])), model
    if method == "seasonal_naive":
        forecast = seasonal_naive_forecast(train, dates)
        return add_balance(forecast, float(train["Saldo_Bancario"].iloc[-1])), None
    raise ValueError(f"Método no reconocido: {method}")


def evaluate_window(data: pd.DataFrame, start: pd.Timestamp, split_name: str) -> list[dict]:
    end = start + pd.Timedelta(days=HORIZON - 1)
    train = data[data["Fecha"] < start].copy()
    actual = data[data["Fecha"].between(start, end)].copy()
    if len(actual) != HORIZON:
        raise ValueError(f"{split_name} no contiene {HORIZON} observaciones")

    rows = []
    for method in ("seasonal_naive", "gradient_boosting"):
        forecast, _ = forecast_method(method, train, pd.DatetimeIndex(actual["Fecha"]))
        metrics = evaluate_forecast(actual, forecast)
        rows.append(
            {
                "split": split_name,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "method": method,
                **metrics,
            }
        )
    return rows


def create_scenarios(base: pd.DataFrame, opening_balance: float) -> pd.DataFrame:
    result = base[["Fecha"]].copy()
    assumptions = {
        "Base": (1.00, 1.00),
        "Adverso": (0.90, 1.05),
        "Favorable": (1.05, 0.98),
    }
    for name, (collections_factor, operating_factor) in assumptions.items():
        flow = (
            base["Cobros_Previstos"] * collections_factor
            - base["Pagos_Operativos_Previstos"] * operating_factor
            - base["Pagos_Programados"]
        )
        result[f"Flujo_{name}"] = flow
        result[f"Saldo_{name}"] = opening_balance + flow.cumsum()
    return result


def plot_test(
    actual: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(actual["Fecha"], actual["Saldo_Bancario"], label="Saldo real", color="#111827", linewidth=2)
    ax.plot(
        forecasts["seasonal_naive"]["Fecha"],
        forecasts["seasonal_naive"]["Saldo_Previsto"],
        label="Baseline estacional",
        color="#9ca3af",
        linestyle="--",
    )
    ax.plot(
        forecasts["gradient_boosting"]["Fecha"],
        forecasts["gradient_boosting"]["Saldo_Previsto"],
        label="Gradient boosting",
        color="#2563eb",
    )
    ax.set_title("Evaluación final: saldo reconstruido durante 60 días")
    ax.set_ylabel("Saldo (€)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scenarios(scenarios: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {"Base": "#2563eb", "Adverso": "#dc2626", "Favorable": "#16a34a"}
    for scenario, color in colors.items():
        ax.plot(scenarios["Fecha"], scenarios[f"Saldo_{scenario}"], label=scenario, color=color)
    ax.axhline(0, color="#111827", linestyle=":", linewidth=1)
    ax.set_title("Escenarios de liquidez a 60 días")
    ax.set_ylabel("Saldo previsto (€)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    validation_rows: list[dict] = []
    for start in VALIDATION_STARTS:
        validation_rows.extend(evaluate_window(data, pd.Timestamp(start), f"validation_{start}"))
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(REPORTS_DIR / "backtest_metrics.csv", index=False)

    average_validation = (
        validation.groupby("method", as_index=False)["balance_mae"].mean().sort_values("balance_mae")
    )
    selected_method = str(average_validation.iloc[0]["method"])

    test_end = TEST_START + pd.Timedelta(days=HORIZON - 1)
    train_test = data[data["Fecha"] < TEST_START].copy()
    actual_test = data[data["Fecha"].between(TEST_START, test_end)].copy()
    test_forecasts: dict[str, pd.DataFrame] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    for method in ("seasonal_naive", "gradient_boosting"):
        forecast, _ = forecast_method(method, train_test, pd.DatetimeIndex(actual_test["Fecha"]))
        test_forecasts[method] = forecast
        test_metrics[method] = evaluate_forecast(actual_test, forecast)

    plot_test(actual_test, test_forecasts, FIGURES_DIR / "test_balance_comparison.png")

    future_dates = pd.date_range(data["Fecha"].max() + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    final_model = CashFlowForecaster().fit(data)
    if selected_method == "gradient_boosting":
        base_forecast = final_model.predict(future_dates)
    else:
        base_forecast = seasonal_naive_forecast(data, future_dates)
    scenarios = create_scenarios(base_forecast, float(data["Saldo_Bancario"].iloc[-1]))
    scenarios.to_csv(REPORTS_DIR / "forecast_scenarios_60d.csv", index=False)
    plot_scenarios(scenarios, FIGURES_DIR / "forecast_scenarios_60d.png")

    artifact = {
        "model": final_model,
        "selected_method": selected_method,
        "training_end": data["Fecha"].max().date().isoformat(),
        "horizon_days": HORIZON,
    }
    joblib.dump(artifact, ARTIFACTS_DIR / "cash_flow_forecaster.joblib")

    report = {
        "selection_rule": "Menor MAE medio del saldo en tres ventanas de validación de 60 días",
        "selected_method": selected_method,
        "validation_average_balance_mae": {
            row["method"]: round(float(row["balance_mae"]), 2)
            for row in average_validation.to_dict(orient="records")
        },
        "test_period": {
            "start": TEST_START.date().isoformat(),
            "end": test_end.date().isoformat(),
            "used_once_after_selection": True,
        },
        "test_metrics": {
            method: {name: round(float(value), 4) for name, value in metrics.items()}
            for method, metrics in test_metrics.items()
        },
        "scenario_assumptions": {
            "base": "Previsión seleccionada sin ajustes",
            "adverse": "Cobros -10 % y pagos operativos +5 %",
            "favourable": "Cobros +5 % y pagos operativos -2 %",
        },
        "limitations": [
            "Datos completamente sintéticos",
            "Los pagos programados se consideran conocidos y no se modelizan",
            "Los escenarios son supuestos ilustrativos, no probabilidades calibradas",
        ],
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
