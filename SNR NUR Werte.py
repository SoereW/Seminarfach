import itertools
import pathlib
import csv
import cv2
import numpy as np
import scipy.ndimage
import tqdm
import dito

def process_video(video_path):
    # Input
    print(f"Processing '{video_path.name}'...")
    video = cv2.VideoCapture(str(video_path))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # SNR calculation settings
    peak_radius = 12
    background_radius = 64
    default_background_value = 1.0  # Standardwert, falls Hintergrund immer noch 0 ist

    # Prepare CSV file
    results_path = video_path.parent.joinpath("results", f"{video_path.stem}_snr.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # CSV headers
        writer.writerow([
            "Frame", "Time (s)",
            "Peak Blue", "Background Blue", "SNR Blue",
            "Peak Green", "Background Green", "SNR Green",
            "Peak Red", "Background Red", "SNR Red",
            "Star Size", "Half-width Ratio"
        ])

        # Initialize previous background signals to None and star size to a large value
        previous_background_signals = None
        min_star_size = float('inf')

        # For each frame
        for n_frame in tqdm.tqdm(itertools.count(), total=frame_count):
            ret, image = video.read()
            if not ret:
                break  # End of video

            image_gray = dito.as_gray(image)

            # Gauss + tophat
            image_median = dito.median_blur(image_gray, 5)
            image_gauss = dito.gaussian_blur(image_median, 3.0)
            image_tophat = dito.tophat(image_gauss, size=2 * peak_radius + 1)

            # Locate peak and peak signal
            peak_y, peak_x = np.unravel_index(np.argmax(image_tophat), image_tophat.shape)
            peak_image = image[
                         max(0, peak_y - peak_radius):min(image_gray.shape[0], peak_y + peak_radius + 1),
                         max(0, peak_x - peak_radius):min(image_gray.shape[1], peak_x + peak_radius + 1),
                         :,
                         ]
            peak_signals = np.max(peak_image, axis=(0, 1))

            # Get background patch and signals
            background_image = image[
                               max(0, peak_y - background_radius):min(image_gray.shape[0],
                                                                      peak_y + background_radius + 1),
                               max(0, peak_x - background_radius):min(image_gray.shape[1],
                                                                      peak_x + background_radius + 1),
                               :,
                               ]
            background_signals = np.median(background_image, axis=(0, 1))

            # Use previous background if current background is zero
            if previous_background_signals is not None:
                for i in range(3):
                    if background_signals[i] == 0:
                        background_signals[i] = previous_background_signals[i]

            # If still zero, use the default background value
            background_signals[background_signals == 0] = default_background_value

            # Update previous background signals
            previous_background_signals = background_signals

            # Channel-wise SNR values
            channel_snrs = peak_signals / background_signals

            # Calculate time in seconds
            time_in_seconds = n_frame / fps

            # Calculate the star size using the Full Width at Half Maximum (FWHM)
            star_size = np.sum(image_tophat > (0.5 * np.max(image_tophat)))

            # Update minimum star size if the current size is smaller
            if star_size < min_star_size:
                min_star_size = star_size

            # Calculate the half-width ratio
            half_width_ratio = star_size / min_star_size

            # Save all values to CSV
            writer.writerow([
                n_frame, time_in_seconds,
                peak_signals[0], background_signals[0], channel_snrs[0],
                peak_signals[1], background_signals[1], channel_snrs[1],
                peak_signals[2], background_signals[2], channel_snrs[2],
                star_size, half_width_ratio
            ])

    print(f"Finished processing '{video_path.name}'. SNR data saved to '{results_path}'.")


def main():
    video_base_path = pathlib.Path(r"C:\Users\soere\OneDrive\Dokumente\Schule\Seminarfachprojekt\Alle Messungen\PolarsternMorgen2.9.24")
    video_paths = video_base_path.glob("PolarsternSonnenaufgang7.43.avi")
    for video_path in video_paths:
        process_video(video_path=video_path)


if __name__ == '__main__':
    main()
