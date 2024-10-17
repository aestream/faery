from ._base import Colormap

colormap = Colormap.diverging_from_triplet(
    start="#4285F4",
    middle="#191919",
    end="#F4B400",
    half_samples=128,
)
