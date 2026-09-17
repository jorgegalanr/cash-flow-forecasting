"""Genera un escenario sintético y reproducible de tesorería diaria."""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
START_DATE = "2021-01-01"
END_DATE = "2026-02-28"
OPENING_BALANCE = 3_200_000.0


def scheduled_payments(dates: pd.DatetimeIndex) -> np.ndarray:
    """Devuelve pagos conocidos por calendario (importes positivos)."""
    payments = np.zeros(len(dates), dtype=float)
    day = dates.day
    month = dates.month

    payments += np.where(day == 28, 350_000, 0)  # nóminas
    payments += np.where(day == 5, 1_200_000, 0)  # deuda o leasing
    payments += np.where(np.isin(month, [1, 4, 7, 10]) & (day == 20), 400_000, 0)
    payments += np.where((month == 6) & (day == 30), 1_000_000, 0)
    return payments


def generate_cash_flow_data(
    start: str = START_DATE,
    end: str = END_DATE,
    opening_balance: float = OPENING_BALANCE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Crea datos ficticios para una empresa multisede con cobros recurrentes."""
    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start=start, end=end, freq="D")
    day = dates.day.to_numpy()
    month = dates.month.to_numpy()
    weekday = dates.dayofweek.to_numpy()
    years_from_start = (dates.year - dates.year.min()).to_numpy()

    recurring_month = np.isin(month, [1, 2, 3, 4, 5, 6, 9, 10, 11, 12])
    collection_window = day <= 5
    summer = np.isin(month, [7, 8])
    weekend = weekday >= 5

    growth = 1 + 0.018 * years_from_start
    collections_mean = np.where(
        recurring_month & collection_window,
        450_000,
        np.where(summer, np.where(weekend, 95_000, 65_000), 3_000),
    )
    collections_sd = np.where(
        recurring_month & collection_window,
        50_000,
        np.where(summer, np.where(weekend, 10_000, 8_000), 500),
    )
    collections = np.maximum(rng.normal(collections_mean * growth, collections_sd), 0)

    operating_mean = np.where(summer, 25_000, 15_000) * (1 + 0.012 * years_from_start)
    operating_sd = np.where(summer, 2_000, 1_500)
    operating_payments = np.maximum(rng.normal(operating_mean, operating_sd), 0)
    programmed_payments = scheduled_payments(dates)

    net_cash_flow = collections - operating_payments - programmed_payments
    bank_balance = opening_balance + np.cumsum(net_cash_flow)

    return pd.DataFrame(
        {
            "Fecha": dates,
            "Cobros": collections,
            "Pagos_Operativos": operating_payments,
            "Pagos_Programados": programmed_payments,
            "Flujo_Caja": net_cash_flow,
            "Saldo_Bancario": bank_balance,
        }
    ).round(2)


def main() -> None:
    output_path = Path("data/tesoreria_sintetica.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_cash_flow_data().to_csv(output_path, index=False)
    print(f"Datos sintéticos guardados en {output_path}")


if __name__ == "__main__":
    main()
