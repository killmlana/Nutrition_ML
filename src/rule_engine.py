def check_anemia(hb):
    return hb < 11


def check_muac(muac):
    """MUAC-based acute malnutrition screening (WHO cutoffs).
    Returns: 'sam' if <115mm, 'mam' if <125mm, 'normal' otherwise."""
    if muac < 115:
        return "sam"
    if muac < 125:
        return "mam"
    return "normal"
