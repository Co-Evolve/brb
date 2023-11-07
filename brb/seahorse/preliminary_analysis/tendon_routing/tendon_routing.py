from typing import Dict, List

import numpy as np

from brb.seahorse.morphology.utils import get_all_tendon_start_and_stop_segment_indices


def get_max_number_of_routing_holes(
        num_segments: int,
        tendon_segment_span: int
        ) -> List[int]:
    return num_segments - tendon_segment_span


def get_overlap_amount_per_tendon(
        num_segments: int,
        tendon_segment_span: int
        ) -> Dict[int, int]:
    tendon_indices = get_all_tendon_start_and_stop_segment_indices(
            total_num_segments=num_segments, segment_span=tendon_segment_span
            )
    num_overlap_per_tendon = {stop_index: 1 + min(start_index, tendon_segment_span) for start_index, stop_index in
                              tendon_indices}
    return num_overlap_per_tendon


def get_rotation_for_translation(
        B: float,
        r: float
        ) -> float:
    return np.arccos((2 - (B / r) ** 2) / 2) / np.pi * 180


def get_translation(
        radius: float,
        beta: float
        ) -> float:
    ALPHA = 26.04 / 180 * np.pi
    P = np.array([233, 24 - 42.5])
    R = radius * np.array([np.cos(ALPHA), np.sin(ALPHA)])
    T = np.linalg.norm(R - P)

    beta = ALPHA + (beta / 180 * np.pi)
    RR = radius * np.array([np.cos(beta), np.sin(beta)])
    TT = np.linalg.norm(RR - P)

    displacement = TT - T
    return displacement


def get_radius_for_translation(
        t: float,
        beta: float
        ) -> tuple[float, float]:
    radius_options = np.linspace(25, 150, num=5000)
    minimum_error = np.inf
    best_radius = None

    for r in radius_options:
        translation = get_translation(radius=r, beta=beta)
        error = np.abs(t - translation)
        if error < minimum_error:
            best_radius = r
            minimum_error = error
    return best_radius, minimum_error


def get_beta_and_radii_for_translations(
        translations: np.ndarray
        ) -> tuple[float, float]:
    beta_options = np.arange(0, 150)

    minimum_total_error = np.inf
    best_beta = None
    for beta_option in beta_options:
        total_error = 0
        for translation in translations:
            _, error = get_radius_for_translation(t=translation, beta=beta_option)
            total_error += error

        if total_error < minimum_total_error:
            best_beta = beta_option
            minimum_total_error = total_error

    return best_beta


if __name__ == '__main__':
    # print(get_max_number_of_routing_holes(
    #     num_segments=20, tendon_segment_span=3
    #     ))
    print(get_overlap_amount_per_tendon(num_segments=20, tendon_segment_span=7))
    # print(get_rotation_for_translation(B=20, r=15))


    target_translations = 20 * np.arange(1, 9)
    beta = get_beta_and_radii_for_translations(translations=target_translations)
    print(f"Beta: {beta}")
    for translation in target_translations:
        radius, error = get_radius_for_translation(t=translation, beta=beta)
        print(f"Translation: {translation} \t -> radius: {radius:3f} \t || error: {error:3f}")


