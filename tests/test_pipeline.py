import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cashflow_forecasting.modeling import CashFlowForecaster, add_balance
from generador_datos import generate_cash_flow_data, scheduled_payments


def test_generated_data_is_reproducible_and_balanced():
    first = generate_cash_flow_data(start="2023-01-01", end="2024-12-31")
    second = generate_cash_flow_data(start="2023-01-01", end="2024-12-31")

    pd.testing.assert_frame_equal(first, second)
    reconstructed = (
        first["Cobros"] - first["Pagos_Operativos"] - first["Pagos_Programados"]
    )
    np.testing.assert_allclose(first["Flujo_Caja"], reconstructed, atol=0.02)


def test_programmed_payments_follow_calendar():
    dates = pd.DatetimeIndex(["2026-01-05", "2026-01-20", "2026-01-28", "2026-02-10"])
    payments = scheduled_payments(dates)

    assert payments.tolist() == [1_200_000, 400_000, 350_000, 0]


def test_forecast_has_expected_horizon_and_balance():
    data = generate_cash_flow_data(start="2021-01-01", end="2025-12-31")
    future = pd.date_range("2026-01-01", periods=60, freq="D")
    model = CashFlowForecaster().fit(data)
    forecast = add_balance(model.predict(future), float(data["Saldo_Bancario"].iloc[-1]))

    assert len(forecast) == 60
    assert forecast["Fecha"].is_monotonic_increasing
    assert forecast.notna().all().all()
    expected_end = data["Saldo_Bancario"].iloc[-1] + forecast["Flujo_Caja_Previsto"].sum()
    assert np.isclose(forecast["Saldo_Previsto"].iloc[-1], expected_end)
