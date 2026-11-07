"""B2 : alignement statistique des caractéristiques."""


def align_channel_statistics(features, source_mean, source_std, target_mean, target_std, eps=1e-6):
    """Alignement affine par canal. À adapter aux dimensions réelles des tenseurs."""
    return ((features - target_mean) / (target_std + eps)) * source_std + source_mean
