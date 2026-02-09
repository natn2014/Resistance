# Mixing Silver Resistance Predictor

A PySide6-based GUI application for predicting and optimizing silver resistance mixing formulas using online machine learning.

## Features

- **Mix Input Management**: Input resistance values and weights for three components (High, Low, Recycle)
- **Online Learning**: Real-time model training with adjustable learning rate
- **Prediction**: Weighted average predictions with online regression correction
- **Target Solver**: Find optimal weight combinations for desired resistance values
- **Formula Management**: Create and manage multiple mixing formulas
- **Data Visualization**: 
  - KDE comparison charts of actual vs predicted resistance
  - Formula pairplot for multi-formula analysis
- **History Tracking**: CSV-based history for each formula with timestamps
- **Model Summary**: Displays learning rate, MAE, weights, and sample count

## Requirements

- Python 3.10+
- PySide6 >= 6.6
- matplotlib >= 3.8
- seaborn >= 0.13
- pandas >= 2.1

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application in development mode:
```bash
python main.py
```

### Tabs

- **Mixing**: Input mix parameters, compute predictions, update model with actual measurements
- **Model**: View model statistics and KDE chart comparing actual vs predicted resistance
- **Formulas**: Add or switch between different mixing formulas

### Workflow

1. Set resistance values (High, Low, Recycle)
2. Set weights for each component
3. Click **Compute** to see base and adjusted predictions
4. Enter actual measured resistance
5. Click **Update Model** to train the regressor
6. Monitor learning in the Model tab

## Building with PyInstaller

Create a standalone executable with bundled icon:

```bash
pyinstaller --clean --noconfirm --onefile --windowed \
  --icon "omega.ico" \
  --add-data "omega.ico;." \
  main.py
```

Output: `dist/main.exe`

The build includes:
- Taskbar icon (AppUserModelID: MixingSilver.App)
- Window icon
- Bundled resources

## File Structure

```
Mixing_Silver/
├── main.py                    # Main application
├── requirements.txt           # Python dependencies
├── formulas.json              # Stored formula names
├── mix_history_*.csv          # Per-formula history files
├── omega.ico                  # Application icon
└── dist/
    └── main.exe              # Compiled executable
```

## Data Files

- `formulas.json`: List of formula names
- `mix_history_Default.csv`: History for Default formula
- `mix_history_P703.csv`: History for P703 formula
- `mix_history_RG01-R.csv`: History for RG01-R formula

Each history file stores: Timestamp, Resistances, Weights, Predictions, Actual Values, Errors, Model Weights

## Model Details

**Online Regressor**:
- 5-weight linear model with features: [1.0, fraction_a, fraction_b, fraction_c, base_prediction]
- Updates via gradient descent on prediction error
- Learning rate controlled by slider (0.0001 to 0.1)

**Prediction Formula**:
```
base_pred = (r_a * w_a + r_b * w_b + r_c * w_c) / total_weight
adjusted_pred = base_pred + correction
```

## License

Internal use. See `.github/copilot-instructions.md` for details.
