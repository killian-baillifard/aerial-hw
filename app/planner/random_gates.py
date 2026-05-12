import numpy as np

if __name__ == "__main__":

    MIN_RADIUS = 0.5
    MAX_RADIUS = 1.5
    MIN_HEIGHT = 1.0
    MAX_HEIGHT = 1.5
    SECTORS_START = np.deg2rad(45)
    SECTOR_SPAN = np.deg2rad(30)

    for i in range(5):

        min_azimuth = SECTORS_START + 2 * i * SECTOR_SPAN
        max_azimuth = SECTORS_START + (2 * i + 1) * SECTOR_SPAN
        azimuth = min_azimuth + (max_azimuth - min_azimuth) * np.random.random()

        radius = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * np.random.random()
        x = -radius * np.cos(azimuth)
        y = -radius * np.sin(azimuth)
        z = MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * np.random.random()

        min_yaw = min_azimuth - np.pi / 2
        max_yaw = max_azimuth - np.pi / 2
        yaw = min_yaw + (max_yaw - min_yaw) * np.random.random()

        print(f"{x:.3f},{y:.3f},{z:.3f},{yaw:.3f}")
