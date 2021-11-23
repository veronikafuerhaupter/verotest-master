import numpy as np

class Gaussiannoisecreator:

    def gaussian_noise(self, image):

        mean = 0
        sigma = 0.03

        noise = np.random.normal(mean, sigma, np.shape(image))
        mask_overflow_upper = image + noise
        mask_overflow_lower = image + noise
        noise[mask_overflow_upper]
        noise[mask_overflow_lower]
        image += noise
        return image


