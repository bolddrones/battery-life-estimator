# Battery Life Estimator (UAV)

Predict remaining flight time under current load and wind using a simple empirical model you can calibrate from 5–10 past flights.

## What it does

- **Calibrate** a model from your flight logs (payload, speed, wind, flight time).
- **Estimate total flight time** for a given payload & airspeed.
- **Estimate remaining time** based on current pack voltage (via a LiPo voltage→SOC curve).

## Model (MVP)

We fit a linear model on the inverse of total flight time (minutes):

```
1 / T ≈ b0 + b1 * payload_kg + b2 * v_air^2
```

where `v_air = ground_speed_mps + headwind_mps` (tailwind is negative).

Then we estimate **remaining time** as:

```
T_remaining ≈ T_total * SOC(voltage_per_cell)
```

SOC is mapped from an industry-typical LiPo 3.3–4.2 V per cell curve (see code).

> This is deliberately simple, fast, and explainable. You can later add more features (temperature, altitude, prop type) or non-linear terms.

## Quick start

1. Put your flights into a CSV with columns:

   - `flight_time_min` – total flight time (minutes)
   - `payload_kg`
   - `ground_speed_mps`
   - `headwind_mps` – positive=headwind, negative=tailwind

2. Calibrate and save a model:

```
python battery_life_estimator.py calibrate --csv example_flights.csv --out model.json
```

3. Estimate **total** flight time for a new config:

```
python battery_life_estimator.py estimate-total --model model.json   --payload-kg 0.6 --ground-speed-mps 10 --headwind-mps 2
```

4. Estimate **remaining** time using current voltage (per-cell):

```
python battery_life_estimator.py estimate-remaining --model model.json   --payload-kg 0.6 --ground-speed-mps 10 --headwind-mps 2   --voltage-per-cell 3.85
```

## Example data

See `example_flights.csv` for a fake-but-plausible dataset to test the pipeline.

## Notes

- Voltage→SOC mapping uses a smoothed interpolation of typical LiPo discharge (4.20V=100%, ~3.50V≈20%, 3.30V≈0%). Always set your own conservative landing threshold.
- The model assumes steady-state cruise. Hover-heavy profiles or aggressive maneuvers will deviate.
- For quads, `v_air^2` works well as a first-order proxy; for fixed-wing you may want lift/drag terms and throttle% if available.

## License

MIT
