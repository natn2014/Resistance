from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QCheckBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QMessageBox,
)
from PySide6.QtGui import QFont, QFontDatabase, QIcon


@dataclass
class OnlineRegressor:
    learning_rate: float = 0.01
    weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    def predict_correction(self, features: List[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features))

    def update(self, features: List[float], error: float) -> None:
        lr = self.learning_rate
        for i, x in enumerate(features):
            self.weights[i] -= lr * 2.0 * error * x


@dataclass
class MixInputs:
    r_a: float
    r_b: float
    r_c: float
    w_a: float
    w_b: float
    w_c: float


def compute_base_prediction(mix: MixInputs) -> float:
    total_weight = mix.w_a + mix.w_b + mix.w_c
    if total_weight <= 0:
        return 0.0
    return (
        mix.r_a * mix.w_a + mix.r_b * mix.w_b + mix.r_c * mix.w_c
    ) / total_weight


def compute_features(mix: MixInputs, base_pred: float) -> List[float]:
    total_weight = mix.w_a + mix.w_b + mix.w_c
    if total_weight <= 0:
        return [1.0, 0.0, 0.0, 0.0, base_pred]
    f_a = mix.w_a / total_weight
    f_b = mix.w_b / total_weight
    f_c = mix.w_c / total_weight
    return [1.0, f_a, f_b, f_c, base_pred]


def compute_adjusted_prediction(
    mix: MixInputs, regressor: OnlineRegressor
) -> Tuple[float, float]:
    base_pred = compute_base_prediction(mix)
    features = compute_features(mix, base_pred)
    correction = regressor.predict_correction(features)
    return base_pred, base_pred + correction


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mixing Silver Resistance Predictor")
        self.regressor = OnlineRegressor()
        self.formulas_file = Path("formulas.json")
        self.formula_names = self._load_formula_names()
        self.active_formula = self.formula_names[0]
        self.history_file = self._history_file_for(self.active_formula)
        self.error_history: List[float] = []
        self.last_adjusted: float | None = None
        self.last_actual: float | None = None
        self.adjusted_history: List[float] = []
        self.actual_history: List[float] = []
        self.history_data: Dict[str, List[float]] = {
            "R_High": [],
            "R_Low": [],
            "R_Recycle": [],
            "W_High": [],
            "W_Low": [],
            "W_Recycle": [],
            "Base_R": [],
            "Adjusted_R": [],
            "Actual_R": [],
            "Error": [],
        }

        root = QWidget()
        layout = QVBoxLayout(root)

        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        main_layout = QVBoxLayout(self.main_tab)
        self.active_formula_label = QLabel(f"Active formula: {self.active_formula}")
        base_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        base_font.setPointSize(30)
        self.active_formula_label.setFont(base_font)
        main_layout.addWidget(self.active_formula_label)
        main_layout.addWidget(self._build_inputs_group())
        main_layout.addWidget(self._build_prediction_group())
        main_layout.addWidget(self._build_learning_group())
        main_layout.addWidget(self._build_table_group())
        main_layout.addWidget(QLabel("AGC Automotive Thailand Dx"))

        self.model_tab = QWidget()
        model_layout = QVBoxLayout(self.model_tab)
        model_layout.addWidget(self._build_model_tab())
        model_layout.addWidget(QLabel("AGC Automotive Thailand Dx"))

        self.formula_tab = QWidget()
        formula_layout = QVBoxLayout(self.formula_tab)
        formula_layout.addWidget(self._build_formula_tab())
        formula_layout.addStretch()
        formula_layout.addWidget(QLabel("AGC Automotive Thailand Dx"))

        self.tabs.addTab(self.main_tab, "Mixing")
        self.tabs.addTab(self.model_tab, "Model")
        self.tabs.addTab(self.formula_tab, "Formulas")

        layout.addWidget(self.tabs)

        self.setCentralWidget(root)
        self._load_history_from_csv()

    def _build_inputs_group(self) -> QGroupBox:
        group = QGroupBox("Mix Inputs [ป้อนข้อมูลส่วนผสม]")
        grid = QGridLayout(group)

        self.r_a = QDoubleSpinBox()
        self.r_b = QDoubleSpinBox()
        self.r_c = QDoubleSpinBox()
        for spin in (self.r_a, self.r_b, self.r_c):
            spin.setDecimals(3)
            spin.setRange(-10000, 10000)

        self.w_a = QDoubleSpinBox()
        self.w_b = QDoubleSpinBox()
        self.w_c = QDoubleSpinBox()
        for spin in (self.w_a, self.w_b, self.w_c):
            spin.setDecimals(4)
            spin.setRange(0, 100000)
            spin.valueChanged.connect(self._update_total_weight)

        self.lock_a = QCheckBox("Lock [ล็อก]")
        self.lock_b = QCheckBox("Lock [ล็อก]")
        self.lock_c = QCheckBox("Lock [ล็อก]")

        high_group = QGroupBox("High [สีสูง]")
        high_layout = QFormLayout(high_group)
        high_layout.addRow("Resistance High [ค่าความต้านทานสูง]", self.r_a)
        high_layout.addRow("Weight High (g) [น้ำหนักสูง (กรัม)]", self.w_a)
        high_layout.addRow("Lock [ล็อก]", self.lock_a)

        low_group = QGroupBox("Low [สีต่ำ]")
        low_layout = QFormLayout(low_group)
        low_layout.addRow("Resistance Low [ค่าความต้านทานต่ำ]", self.r_b)
        low_layout.addRow("Weight Low (g) [น้ำหนักต่ำ (กรัม)]", self.w_b)
        low_layout.addRow("Lock [ล็อก]", self.lock_b)

        recycle_group = QGroupBox("Recycle [รีไซเคิล]")
        recycle_layout = QFormLayout(recycle_group)
        recycle_layout.addRow("Resistance Recycle [ค่าความต้านทานสีรีไซเคิล]", self.r_c)
        recycle_layout.addRow("Weight Recycle (g) [น้ำหนักรีไซเคิล (กรัม)]", self.w_c)
        recycle_layout.addRow("Lock [ล็อก]", self.lock_c)

        grid.addWidget(high_group, 0, 0)
        grid.addWidget(low_group, 0, 1)
        grid.addWidget(recycle_group, 0, 2)

        self.compute_button = QPushButton("Compute [คำนวณ]")
        self.compute_button.clicked.connect(self.compute_prediction)
        grid.addWidget(self.compute_button, 1, 0, 1, 3)

        self.total_weight_label = QLabel("0.0000")
        grid.addWidget(QLabel("Total Weight (g)"), 2, 0)
        grid.addWidget(self.total_weight_label, 2, 1)
        self._update_total_weight()

        return group

    def _build_prediction_group(self) -> QGroupBox:
        group = QGroupBox("Prediction [พยากรณ์]")
        layout = QFormLayout(group)

        self.base_pred_label = QLabel("-")
        self.adjusted_pred_label = QLabel("-")
        self.error_label = QLabel("-")

        layout.addRow("Base R (weighted) [ค่าความต้านทานฐาน]", self.base_pred_label)
        layout.addRow("Adjusted R [ค่าความต้านทานทำนายสุดท้าย]", self.adjusted_pred_label)
        layout.addRow("Last error [ข้อผิดพลาดล่าสุด]", self.error_label)

        return group

    def _build_learning_group(self) -> QGroupBox:
        group = QGroupBox("Online Learning")
        layout = QGridLayout(group)

        self.actual_r = QDoubleSpinBox()
        self.actual_r.setDecimals(4)
        self.actual_r.setRange(-10000, 10000)

        self.update_button = QPushButton("Update Model [ปรับแบบจำลอง]")
        self.update_button.clicked.connect(self.update_model)

        self.target_r = QDoubleSpinBox()
        self.target_r.setDecimals(4)
        self.target_r.setRange(-10000, 10000)

        self.solve_button = QPushButton("Solve Target [หาน้ำหนัก]")
        self.solve_button.clicked.connect(self.solve_target_weight)

        self.solve_status = QLabel("-")

        self.target_total_weight = QDoubleSpinBox()
        self.target_total_weight.setDecimals(4)
        self.target_total_weight.setRange(0, 10000)

        layout.addWidget(QLabel("Actual Resistance [ค่าความต้านทานจริง]"), 0, 0)
        layout.addWidget(self.actual_r, 0, 1)
        layout.addWidget(self.update_button, 0, 2)

        layout.addWidget(QLabel("Target Adjusted Resistance [ค่าความต้านทานที่ต้องการ]"), 1, 0)
        layout.addWidget(self.target_r, 1, 1)
        layout.addWidget(self.solve_button, 1, 2)

        layout.addWidget(QLabel("Target Total Weight (g) [น้ำหนักรวมเป้าหมาย (กรัม)]"), 2, 0)
        layout.addWidget(self.target_total_weight, 2, 1)

        layout.addWidget(QLabel("Solve status [สถานะการแก้ปัญหา]"), 3, 0)
        layout.addWidget(self.solve_status, 3, 1, 1, 2)

        return group

    def _build_table_group(self) -> QGroupBox:
        group = QGroupBox("History")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(group)

        self.table = QTableWidget(0, 12)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table.setHorizontalHeaderLabels(
            [
                "Timestamp",
                "Resistance High",
                "Resistance Low",
                "Resistance Recycle",
                "Weight High (g)",
                "Weight Low (g)",
                "Weight Recycle (g) ",
                "Base R",
                "Adjusted R",
                "Actual Resistance (Measured)",
                "Error",
                "Model Weights",
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.table)
        return group

    def _build_formula_tab(self) -> QGroupBox:
        group = QGroupBox("Mixing Formulas [สูตรผสม]")
        layout = QFormLayout(group)

        self.formula_selector = QComboBox()
        self.formula_selector.addItems(self.formula_names)
        self.formula_selector.currentTextChanged.connect(self._on_formula_changed)

        self.new_formula_input = QLineEdit()
        self.add_formula_button = QPushButton("Add Formula [เพิ่มสูตรผสม]")
        self.add_formula_button.clicked.connect(self._add_formula)

        layout.addRow("Active formula [สูตรผสมที่ใช้งาน]", self.formula_selector)
        layout.addRow("New formula name [ชื่อสูตรผสมใหม่]", self.new_formula_input)
        layout.addRow("", self.add_formula_button)

        return group

    def _build_model_tab(self) -> QGroupBox:
        group = QGroupBox("Model Summary [สรุปแบบจำลอง]")
        layout = QVBoxLayout(group)
        self.model_layout = layout

        form = QFormLayout()

        self.model_lr_label = QLabel("-")
        self.model_samples_label = QLabel("0")
        self.model_last_error_label = QLabel("-")
        self.model_mae_label = QLabel("-")
        self.model_last_adjusted_label = QLabel("-")
        self.model_last_actual_label = QLabel("-")
        self.model_weights_label = QLabel("-")

        form.addRow("Learning rate", self.model_lr_label)
        form.addRow("Samples", self.model_samples_label)
        form.addRow("Last error", self.model_last_error_label)
        form.addRow("Mean abs error", self.model_mae_label)
        form.addRow("Last adjusted", self.model_last_adjusted_label)
        form.addRow("Last actual", self.model_last_actual_label)
        form.addRow("Weights", self.model_weights_label)

        layout.addLayout(form)

        # Controls section for model adjustment
        controls_group = QGroupBox("Model Controls [การควบคุมแบบจำลอง]")
        controls_layout = QGridLayout(controls_group)

        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(1, 1000)
        self.lr_slider.setValue(100)
        self.lr_slider.valueChanged.connect(self._update_lr_label)

        self.lr_label = QLabel("0.0100")

        self.reset_button = QPushButton("Reset Model [รีเซ็ตแบบจำลอง]")
        self.reset_button.clicked.connect(self.reset_model)

        self.weights_view = QLineEdit()
        self.weights_view.setReadOnly(True)

        controls_layout.addWidget(QLabel("Sensitivity (learning rate) [ความไว (อัตราการเรียนรู้)]"), 0, 0)
        controls_layout.addWidget(self.lr_slider, 0, 1)
        controls_layout.addWidget(self.lr_label, 0, 2)

        controls_layout.addWidget(QLabel("Model weights"), 1, 0)
        controls_layout.addWidget(self.weights_view, 1, 1)
        controls_layout.addWidget(self.reset_button, 1, 2)

        layout.addWidget(controls_group)

        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.ax_corr = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas)

        self.pairplot_button = QPushButton("Show Formula Pairplot [แสดงแผนภาพคู่สูตรผสม]")
        self.pairplot_button.clicked.connect(self._show_formula_pairplot)
        layout.addWidget(self.pairplot_button)

        self._update_lr_label()
        self._refresh_weights_view()
        self._refresh_model_summary()
        return group

    def _get_mix_inputs(self) -> MixInputs:
        return MixInputs(
            r_a=self.r_a.value(),
            r_b=self.r_b.value(),
            r_c=self.r_c.value(),
            w_a=self.w_a.value(),
            w_b=self.w_b.value(),
            w_c=self.w_c.value(),
        )

    def compute_prediction(self) -> None:
        mix = self._get_mix_inputs()
        base_pred, adjusted = compute_adjusted_prediction(mix, self.regressor)

        self.base_pred_label.setText(f"{base_pred:.4f}")
        self.adjusted_pred_label.setText(f"{adjusted:.4f}")

    def update_model(self) -> None:
        mix = self._get_mix_inputs()
        base_pred, adjusted = compute_adjusted_prediction(mix, self.regressor)
        features = compute_features(mix, base_pred)

        actual = self.actual_r.value()
        error = adjusted - actual
        self.regressor.update(features, error)

        self.error_label.setText(f"{error:.4f}")
        self.adjusted_pred_label.setText(f"{adjusted:.4f}")
        self.base_pred_label.setText(f"{base_pred:.4f}")

        self._refresh_weights_view()
        self.error_history.append(error)
        self.last_adjusted = adjusted
        self.last_actual = actual
        self.adjusted_history.append(adjusted)
        self.actual_history.append(actual)
        self._append_history_metrics(mix, base_pred, adjusted, actual, error)
        self._refresh_model_summary()
        self._append_history_row(mix, base_pred, adjusted, actual, error)

    def reset_model(self) -> None:
        self.regressor.weights = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.error_label.setText("-")
        self._refresh_weights_view()
        self.error_history = []
        self.last_adjusted = None
        self.last_actual = None
        self.adjusted_history = []
        self.actual_history = []
        self._reset_history_metrics()
        self._refresh_model_summary()

    def solve_target_weight(self) -> None:
        mix = self._get_mix_inputs()
        target = self.target_r.value()
        locked = [self.lock_a.isChecked(), self.lock_b.isChecked(), self.lock_c.isChecked()]
        variable_indices = [i for i, is_locked in enumerate(locked) if not is_locked]
        target_total = self.target_total_weight.value()

        if len(variable_indices) == 0:
            self.solve_status.setText("Unlock at least one weight to solve. [ปลดล็อกน้ำหนักอย่างน้อยหนึ่งค่าเพื่อแก้ปัญหา]")
            return

        if target_total <= 0:
            target_total = mix.w_a + mix.w_b + mix.w_c

        max_weight = max(self.w_a.maximum(), self.w_b.maximum(), self.w_c.maximum())
        locked_sum = sum(
            w for i, w in enumerate([mix.w_a, mix.w_b, mix.w_c]) if i not in variable_indices
        )
        remaining_total = max(0.0, target_total - locked_sum)

        if len(variable_indices) == 1:
            var_idx = variable_indices[0]
            required_weight = min(max_weight, remaining_total)
            if var_idx == 0:
                self.w_a.setValue(required_weight)
                weight_name = "A"
            elif var_idx == 1:
                self.w_b.setValue(required_weight)
                weight_name = "B"
            else:
                self.w_c.setValue(required_weight)
                weight_name = "C"

            _, adjusted = compute_adjusted_prediction(self._get_mix_inputs(), self.regressor)
            self.solve_status.setText(
                f"Solved weight {weight_name} = {required_weight:.4f}, adjusted = {adjusted:.4f}, error = {abs(adjusted - target):.4f}"
            )
            self.compute_prediction()
            return

        def adjusted_for_weights(wa: float, wb: float, wc: float) -> float:
            mix_candidate = MixInputs(
                r_a=mix.r_a,
                r_b=mix.r_b,
                r_c=mix.r_c,
                w_a=wa,
                w_b=wb,
                w_c=wc,
            )
            _, adjusted = compute_adjusted_prediction(mix_candidate, self.regressor)
            return adjusted

        best_error = float("inf")
        best_weights = (mix.w_a, mix.w_b, mix.w_c)

        if len(variable_indices) == 2:
            i1, i2 = variable_indices
            coarse_step = 0.02
            a = 0.0
            while a <= 1.0 + 1e-9:
                w1 = remaining_total * a
                w2 = remaining_total - w1
                weights = [mix.w_a, mix.w_b, mix.w_c]
                weights[i1] = w1
                weights[i2] = w2
                adjusted = adjusted_for_weights(weights[0], weights[1], weights[2])
                err = abs(adjusted - target)
                if err < best_error:
                    best_error = err
                    best_weights = (weights[0], weights[1], weights[2])
                a += coarse_step

        if len(variable_indices) == 3:
            coarse_step = 0.05
            a = 0.0
            while a <= 1.0 + 1e-9:
                b = 0.0
                while b <= 1.0 - a + 1e-9:
                    c = max(0.0, 1.0 - a - b)
                    wa = remaining_total * a
                    wb = remaining_total * b
                    wc = remaining_total * c
                    adjusted = adjusted_for_weights(wa, wb, wc)
                    err = abs(adjusted - target)
                    if err < best_error:
                        best_error = err
                        best_weights = (wa, wb, wc)
                    b += coarse_step
                a += coarse_step

        self.w_a.setValue(min(max_weight, best_weights[0]))
        self.w_b.setValue(min(max_weight, best_weights[1]))
        self.w_c.setValue(min(max_weight, best_weights[2]))

        _, adjusted = compute_adjusted_prediction(self._get_mix_inputs(), self.regressor)
        self.solve_status.setText(
            f"Solved weights with total {target_total:.4f}, adjusted = {adjusted:.4f}, error = {abs(adjusted - target):.4f}"
        )
        self.compute_prediction()
        return


    def _update_total_weight(self) -> None:
        total = self.w_a.value() + self.w_b.value() + self.w_c.value()
        self.total_weight_label.setText(f"{total:.4f}")

    def _append_history_row(
        self,
        mix: MixInputs,
        base_pred: float,
        adjusted: float,
        actual: float,
        error: float,
    ) -> None:
        timestamp = datetime.now().strftime("%Y/%m/%d %H:%M")
        model_weights = ", ".join(f"{w:.4f}" for w in self.regressor.weights)
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            timestamp,
            mix.r_a,
            mix.r_b,
            mix.r_c,
            mix.w_a,
            mix.w_b,
            mix.w_c,
            base_pred,
            adjusted,
            actual,
            error,
            model_weights,
        ]
        for col, val in enumerate(values):
            if isinstance(val, str):
                item = QTableWidgetItem(val)
            else:
                item = QTableWidgetItem(f"{val:.4f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)

        self._save_history_row(values)

    def _save_history_row(self, values: List[object]) -> None:
        file_exists = self.history_file.exists()
        with self.history_file.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(
                    [
                        "Timestamp",
                        "Resistance High",
                        "Resistance Low",
                        "Resistance Recycle",
                        "Weight High (g)",
                        "Weight Low (g)",
                        "Weight Recycle (g)",
                        "Base R",
                        "Adjusted R",
                        "Actual Resistance",
                        "Error",
                        "Model Weights",
                    ]
                )
            writer.writerow(values)

    def _append_history_row_from_csv(self, row_values: List[str]) -> None:
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)
        for col, val in enumerate(row_values[:12]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_index, col, item)

    def _load_history_from_csv(self) -> None:
        if not self.history_file.exists():
            self._reset_history_state(clear_table=True)
            return

        try:
            with self.history_file.open("r", newline="", encoding="utf-8") as csvfile:
                rows = list(csv.reader(csvfile))
        except OSError:
            return

        if len(rows) < 2:
            self._reset_history_state(clear_table=True)
            return

        data_rows = [row for row in rows[1:] if len(row) >= 12]
        if not data_rows:
            self._reset_history_state(clear_table=True)
            return

        self.table.setRowCount(0)
        for row in data_rows:
            self._append_history_row_from_csv(row)

        last_row = data_rows[-1]

        self.error_history = []
        self.adjusted_history = []
        self.actual_history = []
        self._reset_history_metrics()
        for row in data_rows:
            try:
                self.error_history.append(float(row[10]))
                self.adjusted_history.append(float(row[8]))
                self.actual_history.append(float(row[9]))
                self.history_data["R_High"].append(float(row[1]))
                self.history_data["R_Low"].append(float(row[2]))
                self.history_data["R_Recycle"].append(float(row[3]))
                self.history_data["W_High"].append(float(row[4]))
                self.history_data["W_Low"].append(float(row[5]))
                self.history_data["W_Recycle"].append(float(row[6]))
                self.history_data["Base_R"].append(float(row[7]))
                self.history_data["Adjusted_R"].append(float(row[8]))
                self.history_data["Actual_R"].append(float(row[9]))
                self.history_data["Error"].append(float(row[10]))
            except ValueError:
                continue

        try:
            self.r_a.setValue(float(last_row[1]))
            self.r_b.setValue(float(last_row[2]))
            self.r_c.setValue(float(last_row[3]))
            self.w_a.setValue(float(last_row[4]))
            self.w_b.setValue(float(last_row[5]))
            self.w_c.setValue(float(last_row[6]))
            self.actual_r.setValue(float(last_row[9]))
            self.last_adjusted = float(last_row[8])
            self.last_actual = float(last_row[9])
        except ValueError:
            return

        weights_text = last_row[11]
        if weights_text:
            parts = [p.strip() for p in weights_text.split(",") if p.strip()]
            if len(parts) == len(self.regressor.weights):
                try:
                    self.regressor.weights = [float(p) for p in parts]
                except ValueError:
                    pass

        self._refresh_weights_view()
        self._update_total_weight()
        self._refresh_model_summary()
        self.compute_prediction()

    def _reset_history_state(self, clear_table: bool = False) -> None:
        self.error_history = []
        self.adjusted_history = []
        self.actual_history = []
        self.last_adjusted = None
        self.last_actual = None
        self._reset_history_metrics()
        if clear_table:
            self.table.setRowCount(0)
        self._refresh_model_summary()

    def _history_file_for(self, formula_name: str) -> Path:
        safe_name = "".join(c for c in formula_name if c.isalnum() or c in "-_ ").strip()
        safe_name = safe_name.replace(" ", "_") or "Default"
        return Path(f"mix_history_{safe_name}.csv")

    def _load_formula_names(self) -> List[str]:
        if not self.formulas_file.exists():
            return ["Default"]
        try:
            data = json.loads(self.formulas_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["Default"]
        if not isinstance(data, list) or not data:
            return ["Default"]
        return [str(name) for name in data if str(name).strip()] or ["Default"]

    def _save_formula_names(self) -> None:
        try:
            self.formulas_file.write_text(
                json.dumps(self.formula_names, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

    def _on_formula_changed(self, name: str) -> None:
        if not name:
            return
        self.active_formula = name
        self.active_formula_label.setText(f"Active formula [สูตรผสมที่ใช้งาน]: {self.active_formula}")
        self.history_file = self._history_file_for(self.active_formula)
        self._load_history_from_csv()

    def _add_formula(self) -> None:
        name = self.new_formula_input.text().strip()
        if not name:
            return
        if name in self.formula_names:
            self.formula_selector.setCurrentText(name)
            return
        self.formula_names.append(name)
        self.formula_selector.addItem(name)
        self.formula_selector.setCurrentText(name)
        self.new_formula_input.clear()
        self._save_formula_names()

    def _update_lr_label(self) -> None:
        lr = self.lr_slider.value() / 10000.0
        self.regressor.learning_rate = lr
        self.lr_label.setText(f"{lr:.4f}")
        if hasattr(self, "model_lr_label"):
            self.model_lr_label.setText(f"{lr:.4f}")

    def _refresh_weights_view(self) -> None:
        weights = ", ".join(f"{w:.4f}" for w in self.regressor.weights)
        self.weights_view.setText(weights)

    def _refresh_model_summary(self) -> None:
        self.model_lr_label.setText(f"{self.regressor.learning_rate:.4f}")
        self.model_samples_label.setText(str(len(self.error_history)))
        if self.error_history:
            last_error = self.error_history[-1]
            mae = sum(abs(e) for e in self.error_history) / len(self.error_history)
            self.model_last_error_label.setText(f"{last_error:.4f}")
            self.model_mae_label.setText(f"{mae:.4f}")
        else:
            self.model_last_error_label.setText("-")
            self.model_mae_label.setText("-")

        if self.last_adjusted is not None:
            self.model_last_adjusted_label.setText(f"{self.last_adjusted:.4f}")
        else:
            self.model_last_adjusted_label.setText("-")

        if self.last_actual is not None:
            self.model_last_actual_label.setText(f"{self.last_actual:.4f}")
        else:
            self.model_last_actual_label.setText("-")

        weights = ", ".join(f"{w:.4f}" for w in self.regressor.weights)
        self.model_weights_label.setText(weights)

        self._refresh_model_charts()

    def _refresh_model_charts(self) -> None:
        df = pd.DataFrame(self.history_data)
        fig = Figure(figsize=(5, 4), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        if len(df) < 2:
            ax.set_title("Actual vs Predicted (need 2+ samples)", color="white")
            ax.text(
                0.5,
                0.5,
                "Not enough data",
                color="white",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self._set_model_canvas(fig)
            return

        required_columns = ["Actual_R", "Adjusted_R"]
        if any(col not in df.columns for col in required_columns):
            ax.set_title("Actual vs Predicted", color="white")
            ax.text(
                0.5,
                0.5,
                "Missing columns",
                color="white",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self._set_model_canvas(fig)
            return

        df = df.dropna(subset=required_columns, how="any")
        if len(df) < 2:
            ax.set_title("Actual vs Predicted", color="white")
            ax.text(
                0.5,
                0.5,
                "Not enough valid rows",
                color="white",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self._set_model_canvas(fig)
            return

        sns.kdeplot(
            data=df,
            x="Actual_R",
            ax=ax,
            label="Actual",
            fill=True,
            alpha=0.35,
            linewidth=2,
        )
        sns.kdeplot(
            data=df,
            x="Adjusted_R",
            ax=ax,
            label="Predicted",
            fill=True,
            alpha=0.35,
            linewidth=2,
        )
        ax.set_title("Actual vs Predicted (KDE)", color="white")
        ax.set_xlabel("Resistance", color="white")
        ax.set_ylabel("Density", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        legend = ax.legend(facecolor="black", edgecolor="white")
        for text in legend.get_texts():
            text.set_color("white")

        self._set_model_canvas(fig)

    def _set_model_canvas(self, figure: Figure) -> None:
        if hasattr(self, "canvas") and self.canvas is not None:
            self.model_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
        self.figure = figure
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._attach_canvas_resize_handler(self.canvas, self.figure)
        self.model_layout.addWidget(self.canvas)

    def _attach_canvas_resize_handler(self, canvas: FigureCanvas, figure: Figure) -> None:
        original_resize_event = canvas.resizeEvent

        def _on_resize(event) -> None:  # type: ignore[override]
            dpi = figure.get_dpi()
            width = max(1, event.size().width()) / dpi
            height = max(1, event.size().height()) / dpi
            figure.set_size_inches(width, height, forward=False)
            figure.tight_layout()
            original_resize_event(event)
            canvas.draw_idle()

        canvas.resizeEvent = _on_resize  # type: ignore[assignment]

    def _load_formula_pairplot_data(self) -> pd.DataFrame:
        column_map = {
            "Resistance High": "R_High",
            "Resistance Low": "R_Low",
            "Resistance Recycle": "R_Recycle",
            "Weight High (g)": "W_High",
            "Weight Low (g)": "W_Low",
            "Weight Recycle (g)": "W_Recycle",
            "Base R": "Base_R",
            "Adjusted R": "Adjusted_R",
            "Actual Resistance": "Actual_R",
            "Actual R": "Actual_R",
            "Error": "Error",
        }

        frames: List[pd.DataFrame] = []
        for formula in self.formula_names:
            history_file = self._history_file_for(formula)
            if not history_file.exists():
                continue
            try:
                df = pd.read_csv(history_file)
            except (OSError, pd.errors.ParserError):
                continue
            if df.empty:
                continue

            df = df.rename(columns=column_map)
            df["Formula"] = formula

            for col in column_map.values():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _show_formula_pairplot(self) -> None:
        df = self._load_formula_pairplot_data()
        if df.empty:
            QMessageBox.information(self, "Formula Pairplot", "No history data found for any formula.")
            return

        plot_columns = ["R_High", "R_Low", "R_Recycle", "Actual_R"]
        available_columns = [col for col in plot_columns if col in df.columns]
        if len(available_columns) < 2:
            QMessageBox.information(
                self,
                "Formula Pairplot",
                "Not enough numeric columns available to build a pairplot.",
            )
            return

        df = df.dropna(subset=available_columns, how="any")
        if df.empty:
            QMessageBox.information(
                self,
                "Formula Pairplot",
                "No valid rows found after filtering missing values.",
            )
            return

        try:
            grid = sns.pairplot(
                df,
                vars=available_columns,
                hue="Formula",
                corner=True,
                diag_kind="kde",
                plot_kws={"alpha": 0.7, "s": 28},
            )
        except Exception:
            grid = sns.pairplot(
                df,
                vars=available_columns,
                hue="Formula",
                corner=True,
                diag_kind="hist",
                plot_kws={"alpha": 0.7, "s": 28},
            )
        grid.fig.suptitle("Formula Pairplot", y=1.02)
        grid.fig.tight_layout()

        if hasattr(self, "pairplot_window") and self.pairplot_window is not None:
            self.pairplot_window.close()

        self.pairplot_window = QMainWindow(self)
        self.pairplot_window.setWindowTitle("Formula Pairplot")
        canvas = FigureCanvas(grid.fig)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(canvas)
        self.pairplot_window.setCentralWidget(container)
        self.pairplot_window.resize(900, 700)
        self.pairplot_window.show()

    def _append_history_metrics(
        self,
        mix: MixInputs,
        base_pred: float,
        adjusted: float,
        actual: float,
        error: float,
    ) -> None:
        self.history_data["R_High"].append(mix.r_a)
        self.history_data["R_Low"].append(mix.r_b)
        self.history_data["R_Recycle"].append(mix.r_c)
        self.history_data["W_High"].append(mix.w_a)
        self.history_data["W_Low"].append(mix.w_b)
        self.history_data["W_Recycle"].append(mix.w_c)
        self.history_data["Base_R"].append(base_pred)
        self.history_data["Adjusted_R"].append(adjusted)
        self.history_data["Actual_R"].append(actual)
        self.history_data["Error"].append(error)

    def _reset_history_metrics(self) -> None:
        for key in self.history_data:
            self.history_data[key] = []


def main() -> None:
    app = QApplication(sys.argv)
    icon_path = resource_path("omega.ico")
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
    _set_windows_app_user_model_id("MixingSilver.App")
    window = MainWindow()
    if icon_path:
        window.setWindowIcon(QIcon(icon_path))
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec())


def resource_path(relative_path: str) -> str | None:
    base_path = None
    if getattr(sys, "frozen", False):
        base_path = getattr(sys, "_MEIPASS", None)
    if not base_path:
        base_path = os.path.abspath(".")
    candidate = os.path.join(base_path, relative_path)
    return candidate if os.path.exists(candidate) else None


def _set_windows_app_user_model_id(app_id: str) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        return


if __name__ == "__main__":
    main()
