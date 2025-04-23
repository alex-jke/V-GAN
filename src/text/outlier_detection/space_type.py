import enum


class SpaceType(enum.Enum):
    FULL_SPACE = "Full Space"
    VGAN = "V-GAN"
    FEATURE_BAGGING = "Feature Bagging"
    RANDOM_GUESS = "Random"
