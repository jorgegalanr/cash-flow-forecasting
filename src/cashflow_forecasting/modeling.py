"""Modelado y evaluación temporal del flujo de caja."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from generador_datos import scheduled_payments


TARGETS = ("Cobros", "Pagos_Operativos")


def calendar_features(dates: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    """Construye variables conocidas en el momento de realizar la previsión."""
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    day_of_year = index.dayofyear.to_numpy()
    day_of_week = index.dayofweek.to_numpy()
    day = index.day.to_numpy()
    month = index.month.to_numpy()

    return pd.DataFrame(
        {
            "trend_days": (index - origin).days,
            "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
            "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
            "year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "day_sin": np.sin(2 * np.pi * day / 31),
            "day_cos": np.cos(2 * np.pi * day / 31),
            "collection_window": (day <= 5).astype(int),
            "summer": np.isin(month, [7, 8]).astype(int),
            "weekend": (day_of_week >= 5).astype(int),
            "month_end": index.is_month_end.astype(int),
        },
        index=index,
    )


@dataclass
class CashFlowForecaster:
    """Predice flujos inciertos y añade después los pagos programados."""

    random_state: int = 42

    def fit(self, data: pd.DataFrame) -> "CashFlowForecaster":
        frame = data.copy()
        frame["Fecha"] = pd.to_datetime(frame["Fecha"])
        self.origin_ = frame["Fecha"].min()
        features = calendar_features(frame["Fecha"], self.origin_)
        self.models_ = {}
        for target in TARGETS:
            model = HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=20,
                l2_regularization=1.0,
                random_state=self.random_state,
            )
            model.fit(features, frame[target])
            self.models_[target] = model
        return self

    def predict(self, dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
        index = pd.DatetimeIndex(pd.to_datetime(dates))
        features = calendar_features(index, self.origin_)
        collections = np.maximum(self.models_["Cobros"].predict(features), 0)
        operating = np.maximum(self.models_["Pagos_Operativos"].predict(features), 0)
        programmed = scheduled_payments(index)
        net = collections - operating - programmed
        return pd.DataFrame(
            {
                "Fecha": index,
                "Cobros_Previstos": collections,
                "Pagos_Operativos_Previstos": operating,
                "Pagos_Programados": programmed,
                "Flujo_Caja_Previsto": net,
            }
        )


def seasonal_naive_forecast(train: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Baseline: reutiliza el mismo día natural del último año disponible."""
    history = train.copy()
    history["Fecha"] = pd.to_datetime(history["Fecha"])
    history = history.set_index("Fecha")
    rows: list[dict[str, float | pd.Timestamp]] = []

    for date in dates:
        previous = date - pd.DateOffset(years=1)
        if previous not in history.index:
            previous = previous - pd.Timedelta(days=1)
        source = (
            history.loc[previous]
            if previous in history.index
            else history.iloc[-365:][list(TARGETS)].median()
        )
        collections = float(source["Cobros"])
        operating = float(source["Pagos_Operativos"])
        programmed = float(scheduled_payments(pd.DatetimeIndex([date]))[0])
        rows.append(
            {
                "Fecha": date,
                "Cobros_Previstos": collections,
                "Pagos_Operativos_Previstos": operating,
                "Pagos_Programados": programmed,
                "Flujo_Caja_Previsto": collections - operating - programmed,
            }
        )
    return pd.DataFrame(rows)


def add_balance(forecast: pd.DataFrame, opening_balance: float) -> pd.DataFrame:
    result = forecast.copy()
    result["Saldo_Previsto"] = opening_balance + result["Flujo_Caja_Previsto"].cumsum()
    return result


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = actual - predicted
    denominator = max(float(np.abs(actual).sum()), 1.0)
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "wape": float(np.abs(errors).sum() / denominator),
    }


def evaluate_forecast(actual: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    merged = actual.merge(forecast, on="Fecha", validate="one_to_one")
    flow = regression_metrics(
        merged["Flujo_Caja"].to_numpy(), merged["Flujo_Caja_Previsto"].to_numpy()
    )
    balance = regression_metrics(
        merged["Saldo_Bancario"].to_numpy(), merged["Saldo_Previsto"].to_numpy()
    )
    return {
        "cash_flow_mae": flow["mae"],
        "cash_flow_rmse": flow["rmse"],
        "cash_flow_wape": flow["wape"],
        "balance_mae": balance["mae"],
        "ending_balance_error": float(
            merged["Saldo_Previsto"].iloc[-1] - merged["Saldo_Bancario"].iloc[-1]
        ),
    }
