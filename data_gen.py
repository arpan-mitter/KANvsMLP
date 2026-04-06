import numpy as np
import pandas as pd
import os
from datetime import datetime

class PhysicsDataGenerator:
    def __init__(self):
        # Constants
        self.G = 9.81
        self.K_DRAG = 0.08
        self.T0 = 288.15   # Sea level Kelvin
        self.L = 0.0065    # Lapse rate
        self.P0 = 101325   # Sea level Pa
        self.R_SPEC = 287.05
        
        # Determine the directory where this script is saved
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def calculate_trajectory(self, t, v0, angle_deg):
        """Calculates 2D motion with atmospheric drag."""
        angle = np.radians(angle_deg)
        x = (v0 * np.cos(angle) / self.K_DRAG) * (1 - np.exp(-self.K_DRAG * t))
        y = ((v0 * np.sin(angle) + self.G / self.K_DRAG) / self.K_DRAG) * \
            (1 - np.exp(-self.K_DRAG * t)) - (self.G * t / self.K_DRAG)
        return x, y

    def get_atmosphere_stats(self, altitude):
        """Calculates ISA model stats based on altitude."""
        temp_k = self.T0 - (self.L * altitude)
        temp_c = temp_k - 273.15
        pressure = self.P0 * (1 - (self.L * altitude) / self.T0)**5.257
        density = pressure / (self.R_SPEC * temp_k)
        return temp_c, pressure, density

    def generate(self, n_samples=1000, v0=80, angle=60, filename='physics_output.csv'):
        t = np.linspace(0, 20, n_samples)
        x, y = self.calculate_trajectory(t, v0, angle)

        # Filter ground impact
        mask = y >= 0
        t, x, y = t[mask], x[mask], y[mask]

        temp_c, pressure, density = self.get_atmosphere_stats(y)

        df = pd.DataFrame({
            'timestamp': t,
            'pos_x_m': x,
            'altitude_m': y,
            'temp_c': temp_c,
            'pressure_pa': pressure,
            'air_density_kgm3': density
        })

        # Final save path construction
        save_path = os.path.join(self.base_dir, filename)
        df.to_csv(save_path, index=False)
        
        print(f"File saved successfully to:\n{save_path}")
        return df

if __name__ == "__main__":
    generator = PhysicsDataGenerator()
    data = generator.generate(n_samples=2000, v0=100, angle=45)
    print(data.head())