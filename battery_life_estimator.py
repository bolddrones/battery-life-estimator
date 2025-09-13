#!/usr/bin/env python3
import argparse, json, math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

# --- LiPo voltage (per cell) to SOC mapping (typical curve, smoothed) ---
# Points are (voltage_per_cell, SOC) with SOC in [0,1].
_LIPO_POINTS = np.array([
    [4.20, 1.00],
    [4.10, 0.90],
    [4.00, 0.80],
    [3.95, 0.70],
    [3.90, 0.60],
    [3.85, 0.50],
    [3.80, 0.40],
    [3.75, 0.32],
    [3.70, 0.26],
    [3.65, 0.20],
    [3.60, 0.14],
    [3.55, 0.08],
    [3.50, 0.04],
    [3.45, 0.02],
    [3.40, 0.01],
    [3.35, 0.005],
    [3.30, 0.0],
])

def soc_from_voltage_per_cell(v: float) -> float:
    """Map LiPo per-cell voltage to state of charge [0..1] via piecewise-linear interpolation.
       Values are clipped to [0,1].
    """
    xs = _LIPO_POINTS[:,0]
    ys = _LIPO_POINTS[:,1]
    v_clipped = max(min(v, xs.max()), xs.min())
    soc = float(np.interp(v_clipped, xs[::-1], ys[::-1]))  # ensure descending xs
    return max(0.0, min(1.0, soc))

@dataclass
class Model:
    b0: float
    b1: float
    b2: float
    features: List[str] = None

    def predict_total_time_min(self, payload_kg: float, ground_speed_mps: float, headwind_mps: float) -> float:
        v_air = ground_speed_mps + headwind_mps  # headwind positive increases airspeed
        inv_T = self.b0 + self.b1*payload_kg + self.b2*(v_air**2)
        inv_T = max(inv_T, 1e-9)  # guardrail
        return 1.0 / inv_T

    def to_json(self) -> str:
        return json.dumps({"b0": self.b0, "b1": self.b1, "b2": self.b2, "features": ["1","payload_kg","v_air^2"]}, indent=2)

    @staticmethod
    def from_json(s: str) -> 'Model':
        d = json.loads(s)
        return Model(d["b0"], d["b1"], d["b2"], d.get("features", ["1","payload_kg","v_air^2"]))


def fit_inverse_time_linear(X: np.ndarray, y_invT: np.ndarray) -> Tuple[float,float,float]:
    """Least squares fit for invT = b0 + b1*x1 + b2*x2; X columns: [payload_kg, v_air^2]."""
    n = X.shape[0]
    A = np.hstack([np.ones((n,1)), X])  # add bias
    # Solve min ||A*b - y||_2
    b, *_ = np.linalg.lstsq(A, y_invT, rcond=None)
    return float(b[0]), float(b[1]), float(b[2])


def cmd_calibrate(args):
    df = pd.read_csv(args.csv)
    required = ["flight_time_min", "payload_kg", "ground_speed_mps", "headwind_mps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Compute features
    v_air = df["ground_speed_mps"].values + df["headwind_mps"].values
    X = np.vstack([df["payload_kg"].values, v_air**2]).T
    # Target: inverse of time
    invT = 1.0 / df["flight_time_min"].values
    b0, b1, b2 = fit_inverse_time_linear(X, invT)
    model = Model(b0, b1, b2)
    with open(args.out, "w") as f:
        f.write(model.to_json())
    # Basic fit diagnostics
    pred_invT = (b0 + b1*X[:,0] + b2*X[:,1])
    pred_T = 1.0 / np.maximum(pred_invT, 1e-9)
    mae = np.mean(np.abs(pred_T - df["flight_time_min"].values))
    mape = 100*np.mean(np.abs((pred_T - df["flight_time_min"].values)/df["flight_time_min"].values))
    print(f"Saved model to {args.out}")
    print(f"Fit: MAE={mae:.2f} min, MAPE={mape:.1f}%")
    print(f"Coeffs: b0={b0:.6f}, b1={b1:.6f}, b2={b2:.6f}")


def cmd_estimate_total(args):
    with open(args.model, "r") as f:
        model = Model.from_json(f.read())
    T = model.predict_total_time_min(args.payload_kg, args.ground_speed_mps, args.headwind_mps)
    print(f"Estimated TOTAL flight time: {T:.2f} minutes")


def cmd_estimate_remaining(args):
    with open(args.model, "r") as f:
        model = Model.from_json(f.read())
    soc = soc_from_voltage_per_cell(args.voltage_per_cell)
    T_total = model.predict_total_time_min(args.payload_kg, args.ground_speed_mps, args.headwind_mps)
    T_rem = soc * T_total
    print(f"SOC from {args.voltage_per_cell:.2f} V/cell ≈ {soc*100:.1f}%")
    print(f"Estimated TOTAL flight time: {T_total:.2f} minutes")
    print(f"Estimated REMAINING flight time: {T_rem:.2f} minutes")

def main():
    p = argparse.ArgumentParser(description="Battery Life Estimator (UAV) - MVP")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="Fit model from a CSV of past flights")
    pc.add_argument("--csv", required=True, help="CSV with columns: flight_time_min,payload_kg,ground_speed_mps,headwind_mps")
    pc.add_argument("--out", required=True, help="Output model.json file")
    pc.set_defaults(func=cmd_calibrate)

    pe = sub.add_parser("estimate-total", help="Estimate total flight time for given payload/speed/wind")
    pe.add_argument("--model", required=True, help="model.json from calibration")
    pe.add_argument("--payload-kg", type=float, required=True)
    pe.add_argument("--ground-speed-mps", type=float, required=True)
    pe.add_argument("--headwind-mps", type=float, required=True, help="Positive=headwind, negative=tailwind")
    pe.set_defaults(func=cmd_estimate_total)

    pr = sub.add_parser("estimate-remaining", help="Estimate remaining time using current per-cell voltage")
    pr.add_argument("--model", required=True, help="model.json from calibration")
    pr.add_argument("--payload-kg", type=float, required=True)
    pr.add_argument("--ground-speed-mps", type=float, required=True)
    pr.add_argument("--headwind-mps", type=float, required=True)
    pr.add_argument("--voltage-per-cell", type=float, required=True, help="Current LiPo voltage per cell, e.g., 3.85")
    pr.set_defaults(func=cmd_estimate_remaining)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
