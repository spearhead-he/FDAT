# lundquist_dialog.py - Dialog for Lundquist fitting parameters

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QRadioButton, QButtonGroup,
                            QFormLayout, QGroupBox, QDialogButtonBox, QMessageBox)
from PyQt5.QtCore import Qt
import numpy as np

class LundquistParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lundquist Flux Rope Parameters")
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Orientation parameters
        orientation_box = QGroupBox("Flux Rope Orientation")
        orientation_layout = QFormLayout()
        
        self.theta_input = QLineEdit("45")  # Default value
        self.phi_input = QLineEdit("90")    # Default value
        
        orientation_layout.addRow("Axis Latitude θ (deg):", self.theta_input)
        orientation_layout.addRow("Axis Longitude φ (deg):", self.phi_input)
        
        orientation_box.setLayout(orientation_layout)
        layout.addWidget(orientation_box)
        
        # Rope parameters
        rope_box = QGroupBox("Flux Rope Parameters")
        rope_layout = QFormLayout()
        
        self.p0_input = QLineEdit("0.3")  # Default value
        self.b0_input = QLineEdit("15")   # Default value
        self.t0_input = QLineEdit("40")   # Default value
        
        rope_layout.addRow("Impact Parameter (normalized):", self.p0_input)
        rope_layout.addRow("Axial Field B₀ (nT):", self.b0_input)
        rope_layout.addRow("Expansion Time (h):", self.t0_input)
        
        rope_box.setLayout(rope_layout)
        layout.addWidget(rope_box)
        
        # Helicity
        helicity_box = QGroupBox("Handedness/Helicity")
        helicity_layout = QHBoxLayout()
        
        self.h_positive = QRadioButton("Right-handed (+1)")
        self.h_negative = QRadioButton("Left-handed (-1)")
        self.h_positive.setChecked(True)  # Default
        
        helicity_layout.addWidget(self.h_positive)
        helicity_layout.addWidget(self.h_negative)
        
        helicity_box.setLayout(helicity_layout)
        layout.addWidget(helicity_box)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setMinimumWidth(400)
        
    def validate_and_accept(self):
        """Validate inputs before accepting"""
        try:
            # Validate theta (latitude)
            theta = float(self.theta_input.text())
            if theta < -90 or theta > 90:
                raise ValueError("Latitude must be between -90° and 90°")
                
            # Validate phi (longitude)
            phi = float(self.phi_input.text())
            if phi < 0 or phi > 360:
                raise ValueError("Longitude must be between 0° and 360°")
                
            # Validate impact parameter
            p0 = float(self.p0_input.text())
            if p0 < -1 or p0 > 1:
                raise ValueError("Impact parameter must be between -1 and 1")
                
            # Validate axial field
            b0 = float(self.b0_input.text())
            if b0 <= 0:
                raise ValueError("Axial field must be positive")
                
            # Validate expansion time
            t0 = float(self.t0_input.text())
            if t0 <= 0:
                raise ValueError("Expansion time must be positive")
                
            # All valid
            self.accept()
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
        
    def get_parameters(self):
        """Return the parameter values as a dictionary"""
        h_value = 1 if self.h_positive.isChecked() else -1
        
        return {
            'theta0': float(self.theta_input.text()),
            'phi0': float(self.phi_input.text()),
            'p0': float(self.p0_input.text()),
            'h': h_value,
            'b0': float(self.b0_input.text()),
            't0': float(self.t0_input.text())
        }