import numpy as np
from skimage import color

class processimage:
    @staticmethod
    def process_image(obs):
        # Handle Gym's tuple return format
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract actual observation from (observation, info)
            
        # Convert to uint8 if needed (for newer Gym versions)
        if obs.dtype != np.uint8:
            if obs.max() <= 1.0:  # Assume normalized float
                obs = (obs * 255).astype(np.uint8)
            else:
                obs = obs.astype(np.uint8)
                
        obs_gray = color.rgb2gray(obs).astype(np.float32)
        obs_gray[84:95, 0:12] = 0
        obs_gray[np.abs(obs_gray - 0.68616) < 0.0001] = 1
        obs_gray[np.abs(obs_gray - 0.75630) < 0.0001] = 1
        return (2 * obs_gray - 1).astype(np.float32)