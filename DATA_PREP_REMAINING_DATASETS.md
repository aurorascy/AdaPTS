# Remaining Forecasting Data Preparation

This file records Weather and ExchangeRate data preparation for Stage 7. No synthetic data was generated.

- Source: dunzane/time-series-dataset HuggingFace mirror of LTSF benchmark data

## Weather

- URL: https://huggingface.co/datasets/dunzane/time-series-dataset/resolve/main/LSF/weather/weather.csv
- Local path: `external_data\forecasting\Weather\Weather.csv`
- Exists: True
- File size: 7235425
- Shape: (52696, 22)
- Columns: ['date', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m�)', 'PAR (�mol/m�/s)', 'max. PAR (�mol/m�/s)', 'Tlog (degC)', 'OT']
- Missing values: 0
- Numeric feature count: 21

### DataReader checks

- H=96: train X/y (36280, 21, 512) / (36280, 21, 96); val X/y (5175, 21, 512) / (5175, 21, 96); test X/y (10444, 21, 512) / (10444, 21, 96); n_features=21
- H=192: train X/y (36184, 21, 512) / (36184, 21, 192); val X/y (5079, 21, 512) / (5079, 21, 192); test X/y (10348, 21, 512) / (10348, 21, 192); n_features=21

## ExchangeRate

- URL: https://huggingface.co/datasets/dunzane/time-series-dataset/resolve/main/LSF/exchange_rate/exchange_rate.csv
- Local path: `external_data\forecasting\ExchangeRate\ExchangeRate.csv`
- Exists: True
- File size: 637800
- Shape: (7588, 9)
- Columns: ['date', '0', '1', '2', '3', '4', '5', '6', 'OT']
- Missing values: 0
- Numeric feature count: 8

### DataReader checks

- H=96: train X/y (4704, 8, 512) / (4704, 8, 96); val X/y (665, 8, 512) / (665, 8, 96); test X/y (1422, 8, 512) / (1422, 8, 96); n_features=8
- H=192: train X/y (4608, 8, 512) / (4608, 8, 192); val X/y (569, 8, 512) / (569, 8, 192); test X/y (1326, 8, 512) / (1326, 8, 192); n_features=8

## Machine-readable summary

```json
{
  "source": "dunzane/time-series-dataset HuggingFace mirror of LTSF benchmark data",
  "datasets": {
    "Weather": {
      "url": "https://huggingface.co/datasets/dunzane/time-series-dataset/resolve/main/LSF/weather/weather.csv",
      "csv": {
        "exists": true,
        "path": "external_data\\forecasting\\Weather\\Weather.csv",
        "size": 7235425,
        "shape": [
          52696,
          22
        ],
        "columns": [
          "date",
          "p (mbar)",
          "T (degC)",
          "Tpot (K)",
          "Tdew (degC)",
          "rh (%)",
          "VPmax (mbar)",
          "VPact (mbar)",
          "VPdef (mbar)",
          "sh (g/kg)",
          "H2OC (mmol/mol)",
          "rho (g/m**3)",
          "wv (m/s)",
          "max. wv (m/s)",
          "wd (deg)",
          "rain (mm)",
          "raining (s)",
          "SWDR (W/m�)",
          "PAR (�mol/m�/s)",
          "max. PAR (�mol/m�/s)",
          "Tlog (degC)",
          "OT"
        ],
        "missing": 0,
        "numeric_feature_columns": [
          "p (mbar)",
          "T (degC)",
          "Tpot (K)",
          "Tdew (degC)",
          "rh (%)",
          "VPmax (mbar)",
          "VPact (mbar)",
          "VPdef (mbar)",
          "sh (g/kg)",
          "H2OC (mmol/mol)",
          "rho (g/m**3)",
          "wv (m/s)",
          "max. wv (m/s)",
          "wd (deg)",
          "rain (mm)",
          "raining (s)",
          "SWDR (W/m�)",
          "PAR (�mol/m�/s)",
          "max. PAR (�mol/m�/s)",
          "Tlog (degC)",
          "OT"
        ],
        "numeric_feature_count": 21,
        "head": [
          {
            "date": "2020-01-01 00:10:00",
            "p (mbar)": 1008.89,
            "T (degC)": 0.71,
            "Tpot (K)": 273.18,
            "Tdew (degC)": -1.33,
            "rh (%)": 86.1,
            "VPmax (mbar)": 6.43,
            "VPact (mbar)": 5.54,
            "VPdef (mbar)": 0.89,
            "sh (g/kg)": 3.42,
            "H2OC (mmol/mol)": 5.49,
            "rho (g/m**3)": 1280.62,
            "wv (m/s)": 1.02,
            "max. wv (m/s)": 1.6,
            "wd (deg)": 224.3,
            "rain (mm)": 0.0,
            "raining (s)": 0.0,
            "SWDR (W/m�)": 0.0,
            "PAR (�mol/m�/s)": 0.0,
            "max. PAR (�mol/m�/s)": 0.0,
            "Tlog (degC)": 11.45,
            "OT": 428.1
          },
          {
            "date": "2020-01-01 00:20:00",
            "p (mbar)": 1008.76,
            "T (degC)": 0.75,
            "Tpot (K)": 273.22,
            "Tdew (degC)": -1.44,
            "rh (%)": 85.2,
            "VPmax (mbar)": 6.45,
            "VPact (mbar)": 5.49,
            "VPdef (mbar)": 0.95,
            "sh (g/kg)": 3.39,
            "H2OC (mmol/mol)": 5.45,
            "rho (g/m**3)": 1280.33,
            "wv (m/s)": 0.43,
            "max. wv (m/s)": 0.84,
            "wd (deg)": 206.8,
            "rain (mm)": 0.0,
            "raining (s)": 0.0,
            "SWDR (W/m�)": 0.0,
            "PAR (�mol/m�/s)": 0.0,
            "max. PAR (�mol/m�/s)": 0.0,
            "Tlog (degC)": 11.51,
            "OT": 428.0
          },
          {
            "date": "2020-01-01 00:30:00",
            "p (mbar)": 1008.66,
            "T (degC)": 0.73,
            "Tpot (K)": 273.21,
            "Tdew (degC)": -1.48,
            "rh (%)": 85.1,
            "VPmax (mbar)": 6.44,
            "VPact (mbar)": 5.48,
            "VPdef (mbar)": 0.96,
            "sh (g/kg)": 3.39,
            "H2OC (mmol/mol)": 5.43,
            "rho (g/m**3)": 1280.29,
            "wv (m/s)": 0.61,
            "max. wv (m/s)": 1.48,
            "wd (deg)": 197.1,
            "rain (mm)": 0.0,
            "raining (s)": 0.0,
            "SWDR (W/m�)": 0.0,
            "PAR (�mol/m�/s)": 0.0,
            "max. PAR (�mol/m�/s)": 0.0,
            "Tlog (degC)": 11.6,
            "OT": 427.6
          },
          {
            "date": "2020-01-01 00:40:00",
            "p (mbar)": 1008.64,
            "T (degC)": 0.37,
            "Tpot (K)": 272.86,
            "Tdew (degC)": -1.64,
            "rh (%)": 86.3,
            "VPmax (mbar)": 6.27,
            "VPact (mbar)": 5.41,
            "VPdef (mbar)": 0.86,
            "sh (g/kg)": 3.35,
            "H2OC (mmol/mol)": 5.37,
            "rho (g/m**3)": 1281.97,
            "wv (m/s)": 1.11,
            "max. wv (m/s)": 1.48,
            "wd (deg)": 206.4,
            "rain (mm)": 0.0,
            "raining (s)": 0.0,
            "SWDR (W/m�)": 0.0,
            "PAR (�mol/m�/s)": 0.0,
            "max. PAR (�mol/m�/s)": 0.0,
            "Tlog (degC)": 11.7,
            "OT": 430.0
          },
          {
            "date": "2020-01-01 00:50:00",
            "p (mbar)": 1008.61,
            "T (degC)": 0.33,
            "Tpot (K)": 272.82,
            "Tdew (degC)": -1.5,
            "rh (%)": 87.4,
            "VPmax (mbar)": 6.26,
            "VPact (mbar)": 5.47,
            "VPdef (mbar)": 0.79,
            "sh (g/kg)": 3.38,
            "H2OC (mmol/mol)": 5.42,
            "rho (g/m**3)": 1282.08,
            "wv (m/s)": 0.49,
            "max. wv (m/s)": 1.4,
            "wd (deg)": 209.6,
            "rain (mm)": 0.0,
            "raining (s)": 0.0,
            "SWDR (W/m�)": 0.0,
            "PAR (�mol/m�/s)": 0.0,
            "max. PAR (�mol/m�/s)": 0.0,
            "Tlog (degC)": 11.81,
            "OT": 432.2
          }
        ]
      },
      "datareader": {
        "96": {
          "n_features": 21,
          "train_X": [
            36280,
            21,
            512
          ],
          "train_y": [
            36280,
            21,
            96
          ],
          "val_X": [
            5175,
            21,
            512
          ],
          "val_y": [
            5175,
            21,
            96
          ],
          "test_X": [
            10444,
            21,
            512
          ],
          "test_y": [
            10444,
            21,
            96
          ]
        },
        "192": {
          "n_features": 21,
          "train_X": [
            36184,
            21,
            512
          ],
          "train_y": [
            36184,
            21,
            192
          ],
          "val_X": [
            5079,
            21,
            512
          ],
          "val_y": [
            5079,
            21,
            192
          ],
          "test_X": [
            10348,
            21,
            512
          ],
          "test_y": [
            10348,
            21,
            192
          ]
        }
      }
    },
    "ExchangeRate": {
      "url": "https://huggingface.co/datasets/dunzane/time-series-dataset/resolve/main/LSF/exchange_rate/exchange_rate.csv",
      "csv": {
        "exists": true,
        "path": "external_data\\forecasting\\ExchangeRate\\ExchangeRate.csv",
        "size": 637800,
        "shape": [
          7588,
          9
        ],
        "columns": [
          "date",
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "OT"
        ],
        "missing": 0,
        "numeric_feature_columns": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "OT"
        ],
        "numeric_feature_count": 8,
        "head": [
          {
            "date": "1990/1/1 0:00",
            "0": 0.7855,
            "1": 1.611,
            "2": 0.861698,
            "3": 0.634196,
            "4": 0.211242,
            "5": 0.006838,
            "6": 0.525486,
            "OT": 0.593
          },
          {
            "date": "1990/1/2 0:00",
            "0": 0.7818,
            "1": 1.61,
            "2": 0.861104,
            "3": 0.633513,
            "4": 0.211242,
            "5": 0.006863,
            "6": 0.523972,
            "OT": 0.594
          },
          {
            "date": "1990/1/3 0:00",
            "0": 0.7867,
            "1": 1.6293,
            "2": 0.86103,
            "3": 0.648508,
            "4": 0.211242,
            "5": 0.006975,
            "6": 0.526316,
            "OT": 0.5973
          },
          {
            "date": "1990/1/4 0:00",
            "0": 0.786,
            "1": 1.637,
            "2": 0.862069,
            "3": 0.650618,
            "4": 0.211242,
            "5": 0.006953,
            "6": 0.523834,
            "OT": 0.597
          },
          {
            "date": "1990/1/5 0:00",
            "0": 0.7849,
            "1": 1.653,
            "2": 0.861995,
            "3": 0.656254,
            "4": 0.211242,
            "5": 0.00694,
            "6": 0.527426,
            "OT": 0.5985
          }
        ]
      },
      "datareader": {
        "96": {
          "n_features": 8,
          "train_X": [
            4704,
            8,
            512
          ],
          "train_y": [
            4704,
            8,
            96
          ],
          "val_X": [
            665,
            8,
            512
          ],
          "val_y": [
            665,
            8,
            96
          ],
          "test_X": [
            1422,
            8,
            512
          ],
          "test_y": [
            1422,
            8,
            96
          ]
        },
        "192": {
          "n_features": 8,
          "train_X": [
            4608,
            8,
            512
          ],
          "train_y": [
            4608,
            8,
            192
          ],
          "val_X": [
            569,
            8,
            512
          ],
          "val_y": [
            569,
            8,
            192
          ],
          "test_X": [
            1326,
            8,
            512
          ],
          "test_y": [
            1326,
            8,
            192
          ]
        }
      }
    }
  }
}
```
