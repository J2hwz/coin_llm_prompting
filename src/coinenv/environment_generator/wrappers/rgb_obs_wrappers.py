import numpy as np
from gymnasium import spaces
from coinenv.environment_generator.wrappers.fog_of_war import FogOfWarWrapper


class OmnidirectionalFogOfWarRGBImgObsWrapper(FogOfWarWrapper):
    """
    Provides a fully-observable RGB image with a line-of-sight based
    "fog of war" effect. Inherits all mask-calculation logic from
    the FogOfWarWrapper.
    """

    def __init__(self, env, tile_size=8, fog_color=100, view_radius=None):
        # Initialize the fog-of-war logic from the parent class
        super().__init__(env, view_radius=view_radius)

        self.tile_size = tile_size
        self.fog_color = (
            (fog_color, fog_color, fog_color)
            if isinstance(fog_color, int)
            else fog_color
        )

        # Define the observation space for the RGB image
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.unwrapped.height * tile_size,
                self.unwrapped.width * tile_size,
                3,
            ),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def reset(self, **kwargs):
        """Resets the environment and the seen mask via the parent class."""
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        """Updates the mask and then applies fog to the RGB image."""
        # This call updates self.seen_mask
        super().observation(obs)

        full_rgb_img = self.unwrapped.get_frame(
            highlight=False, tile_size=self.tile_size
        )
        masked_img = self._apply_fog(full_rgb_img)

        return {**obs, "image": masked_img}

    def _apply_fog(self, rgb_img):
        """Applies fog based on the parent's seen_mask."""
        pixel_seen_mask = np.kron(
            self.seen_mask, np.ones((self.tile_size, self.tile_size), dtype=bool)
        )
        pixel_seen_mask_3d = np.stack([pixel_seen_mask] * 3, axis=-1)
        fog_img = np.full_like(rgb_img, self.fog_color)
        masked_img = np.where(pixel_seen_mask_3d, rgb_img, fog_img)
        return masked_img.astype(np.uint8)
