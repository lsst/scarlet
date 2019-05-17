from functools import partial

import autograd.numpy as np
from proxmin.operators import prox_plus, prox_hard, prox_soft

from . import interpolation
from . import operator
from .bbox import trim
from .cache import Cache


def _fit_pixel_center(morph, center, window=None):
    cy, cx = center
    if window is None:
        window = slice(cy-2, cy+3), slice(cx-2, cx+3)
    _morph = morph[window]
    yx0 = np.array([window[0].start, window[1].start])
    return tuple(np.unravel_index(np.argmax(_morph), _morph.shape) + yx0)


def fit_pixel_center(component, window=None):
    """Use the pixel with the maximum flux as the center

    In case there is a nearby bright neighbor, we only update
    the center within the immediate vascinity of the previous center.
    This allows the center to shift over time, but prevents a large,
    likely unphysical update.

    Parameters
    ----------
    window: tuple of slices
        Slices in y and x of the central region to include in the fit.
        If `window` is `None` then only the 3x3 grid of pixels centered
        on the previous center are used. If it is desired to use the entire
        morphology just set `window=(slice(None), slice(None))`.
    """
    component.pixel_center = _fit_pixel_center(component.morph, component.pixel_center, window)
    return component


def positive_sed(component):
    """Make the SED non-negative
    """
    prox_plus(component.sed, component.step_sed)
    return component


def positive_morph(component):
    """Make the morphology non-negative
    """
    prox_plus(component.morph, component.step_morph)
    return component


def positive(component):
    """Make both the SED and morpholgy non-negative
    """
    prox_plus(component.sed, component.step_sed)
    prox_plus(component.morph, component.step_morph)
    return component


def normalized(component, type='morph_max'):
    """Normalize SED or morphology

    In order to break degeneracies, either the SED or
    morphology should have some form of normalization.
    For consistency this should normally be the same for
    all components in a blend.

    For `type='sed'` the sed matrix is normalized to
    sum to unity.

    For `type='morph'` the morphology matrix is normalized
    to sum to unity.

    For `type='morph_max'` the morphology matrix is
    normalized so that its maximum value is one.
    """
    assert type.lower() in ['sed', 'morph', 'morph_max']
    t = type.lower()

    if t == 'sed':
        norm = component.sed.sum()
        component.sed[:] = component.sed / norm
        component.morph[:] = component.morph * norm
    elif t == 'morph':
        norm = component.morph.sum()
        component.sed[:] = component.sed * norm
        component.morph[:] = component.morph / norm
    elif t == 'morph_max':
        norm = component.morph.max()
        component.sed[:] = component.sed * norm
        component.morph[:] = component.morph / norm
    else:
        raise ValueError("Unrecognized normalization '{0}'".format(type))


def sparse_l0(component, thresh):
    """L0 norm (sparsity) on morphology
    """
    prox_hard(component.morph, component.step_morph, thresh)
    return component


def sparse_l1(component, thresh):
    """L1 norm (sparsity) on morphology
    """
    prox_soft(component.morph, component.step_morph, thresh)
    return component


def _threshold(morph, noise=None):
    """Find the threshold value for a given morphology
    """
    _morph = morph[morph > 0]
    _bins = 50
    # THIS REALLY NEEDS TO BE DONE RIGHT
    # we propagate the noise the source was initialized with, but
    # the value is WAY to high, like 0.1, because it needs to be
    # scaled by the factor the morphology is normalized by, this
    # is future work, for now setting it to some small number
    noise = 1e-4
    # Decrease the bin size for sources with a small number of pixels
    if _morph.size < 500:
        _bins = max(np.int(_morph.size/10), 1)
        if _bins == 1:
            return 0, _bins
    hist, bins = np.histogram(np.log10(_morph).reshape(-1), _bins)
    cutoff = np.where(hist == 0)[0]
    # If all of the pixels are used there is no need to threshold
    if len(cutoff) == 0:
        return 0, _bins
    if noise is not None:
        adjusted = np.min((bins[cutoff[-1]], np.log10(noise)))
    else:
        adjusted = bins[cutoff[-1]]
    return 10**adjusted, _bins


def threshold(component):
    """Set a cutoff threshold for pixels below the noise

    Use the log histogram of pixel values to determine when the
    source is fitting noise. This function works well to prevent
    faint sources from growing large footprints but for large
    diffuse galaxies with a wide range of pixel values this
    does not work as well.
    """
    index = np.argmax(component.sed)
    if component.noise_level is not None:
        noise = component.noise_level[index]
    else:
        noise = None
    thresh, _bins = _threshold(component.morph, noise)
    component.morph[component.morph < thresh] = 0
    bbox = trim(component.morph)
    if not hasattr(component, "bboxes"):
        component.bboxes = {}
    component.bboxes["thresh"] = bbox
    return component


def monotonic(component, pixel_center, use_nearest=False, thresh=0, exact=False, bbox=None):
    """Make morphology monotonically decrease from the center

    Parameters
    ----------
    exact: `bool`
        Whether to use the (very slow) exact monotonicity proximal operator
        (`exact` is `True`) or a projection operator to the monotonic space.

    See `~scarlet.operator.prox_monotonic`
    for a description of the other parameters.
    """
    if bbox is not None:
        # Only apply monotonicity to the pixels inside the bounding box
        morph = component.morph[bbox.slices]
        shape = morph.shape
        if shape[0] <= 1 or shape[1] <= 1:
            return component
        center = pixel_center[0]-bbox.bottom, pixel_center[1]-bbox.left
    else:
        morph = component.morph
        shape = component.shape[-2:]
        center = pixel_center
    morph = morph.copy()

    prox_name = "update.monotonic"
    key = (shape, center, use_nearest, thresh, exact)
    # The creation of this operator is expensive,
    # so load it from memory if possible.
    try:
        prox = Cache.check(prox_name, key)
    except KeyError:
        if not exact:
            prox = operator.prox_strict_monotonic(shape, use_nearest=use_nearest,
                                                  thresh=thresh, center=center)
        else:
            # Exact monotonicy still needs to be tested and implmented with v0.5.
            raise NotImplementedError("Exact monotonicity is not currently supported")
            # cone method for monotonicity: exact but VERY slow
            G = operator.getRadialMonotonicOp(shape, useNearest=use_nearest).L.toarray()
            prox = partial(operator.prox_cone, G=G)
        Cache.set(prox_name, key, prox)

    step_size = component.step_morph
    prox(morph, step_size)
    if bbox is not None:
        component.morph[bbox.slices] = morph
    else:
        component.morph[:] = morph
    return component


def psf_weighted_centroid(component):
    # Extract the arrays from the component
    morph = component.morph
    psf = component.psf

    py, px = component.pixel_center
    if psf is None:
        pindy, pindx = np.indices((5, 5))
        psf = np.exp(-1/2*((pindx-2)**2 + (pindy-2)**2))

    # Determine the width of the psf
    psf_shape_y, psf_shape_x = psf.shape
    # These are all the same value because psf are square and odd, but using
    # multiple variable names to clarify things
    x_rad = y_rad = psf_peak_y = psf_peak_x = psf_shape_x//2

    # Determine the overlapping coordinates of the morphology image and the psf
    # this is needed if the psf image, centered at the peak pixel location goes
    # off the edge of the psf array
    y_morph_ext = np.arange(morph.shape[0])
    x_morph_ext = np.arange(morph.shape[1])

    # calculate the offset in the morph frame such that the center pixel aligns
    # with the psf coordindate frame, where the psf peak is in the center
    y_morph_ext = y_morph_ext - py
    x_morph_ext = x_morph_ext - px

    # Compare the two end points, and take whichever is the smaller radius
    # use that to select the entries that are equal to or below that radius
    # Find the minimum between the end points (if the psf goes off the edge,
    # and the radius of the psf
    rad_endpoints_y = np.min((abs(y_morph_ext[0]), abs(y_morph_ext[-1]), y_rad))
    rad_endpoints_x = np.min((abs(x_morph_ext[0]), abs(x_morph_ext[-1]), x_rad))
    trimmed_y_range = y_morph_ext[abs(y_morph_ext) <= rad_endpoints_y]
    trimmed_x_range = x_morph_ext[abs(x_morph_ext) <= rad_endpoints_x]

    psf_y_range = trimmed_y_range + psf_peak_y
    psf_x_range = trimmed_x_range + psf_peak_x

    morph_y_range = trimmed_y_range + py
    morph_x_range = trimmed_x_range + px

    # Use these arrays to get views of the morphology and psf
    morph_view = morph[morph_y_range][:, morph_x_range]
    psf_view = psf[psf_y_range][:, psf_x_range]

    morph_view_weighted = morph_view*psf_view
    morph_view_weighted_sum = np.sum(morph_view_weighted)
    # build the indices to use the the centroid calculation
    indy, indx = np.indices(psf_view.shape)
    first_moment_y = np.sum(indy*morph_view_weighted) / morph_view_weighted_sum
    first_moment_x = np.sum(indx*morph_view_weighted) / morph_view_weighted_sum
    # build the offset to go from psf_view frame to psf frame to morph frame
    # aka move the peak back by the radius of the psf width adusted for the
    # minimum point in the view
    offset = (morph_y_range[0], morph_x_range[0])

    whole_pixel_center = np.round((first_moment_y, first_moment_x))
    dy, dx = whole_pixel_center - (first_moment_y, first_moment_x)
    morph_pixel_center = tuple((whole_pixel_center + offset).astype(int))
    component.pixel_center = morph_pixel_center
    component.shift = (dy, dx)
    return component


def symmeterize_kspace(component, bbox=None, pad_multiplier=1):
    dy, dx = component.shift
    cy, cx = component.pixel_center
    if bbox is not None:
        morph = component.morph[bbox.slices]
        slices = bbox.slices
        cy = cy - bbox.bottom
        cx = cx - bbox.left
    else:
        morph = component.morph
        slices = (slice(0, morph.shape[0], 1), slice(0, morph.shape[1], 1))

    result = operator.uncentered_operator(morph, wraped_symmeterized_kspace,
                                          center=(cy, cx), dy=dy, dx=dx,
                                          pad_multiplier=pad_multiplier)
    component.morph[slices] = result
    return component


def wraped_symmeterized_kspace(morph, dy, dx, pad_multiplier):
    # Record the morph shape
    shape = morph.shape
    morph = morph.astype(np.float64)
    # This may need turned in to a float if there is weird behavior
    padding = np.max(morph.shape)//2*pad_multiplier
    edges = ((padding, padding), (padding, padding))
    corner = (padding, padding)
    zeroMask = morph <= 0
    morph[zeroMask] = 0
    morph = np.pad(morph, edges, 'constant')
    freq_x = np.fft.fftfreq(morph.shape[1])
    freq_y = np.fft.fftfreq(morph.shape[0])

    # Transform to k space
    kmorph = np.fft.fft2(np.fft.ifftshift(morph))

    # Shift the signal to recenter it, negative because math is opposite from
    # pixel direction
    shifter = np.outer(np.exp(-1j*2*np.pi*freq_y*-(dy)),
                       np.exp(-1j*2*np.pi*freq_x*-(dx)))

    kmorph_shift = kmorph*shifter
    # symmeterize
    kmorph_shift_sym = np.real(kmorph_shift)

    # Shift back
    kmorph_shift_back = kmorph_shift_sym*np.outer(np.exp(-1j*2*np.pi*freq_y*(dy)),
                                                  np.exp(-1j*2*np.pi*freq_x*(dx)))

    # Transform to real space
    sym_morph = np.fft.fftshift(np.fft.ifft2(kmorph_shift_back))
    # Return the unpadded transform
    sym_morph_unpad = np.real(sym_morph[corner[0]:corner[0]+shape[0], corner[1]:corner[1]+shape[1]])
    sym_morph_unpad[zeroMask] = 0
    assert sym_morph_unpad.shape == shape
    return sym_morph_unpad.astype(float)


def translation(component, direction=1, kernel=interpolation.lanczos, padding=3):
    """Shift the morphology by a given amount
    """
    dy, dx = component.shift
    dy *= direction
    dx *= direction
    _kernel, _, _ = interpolation.get_separable_kernel(dy, dx, kernel=kernel)
    component.morph[:] = interpolation.fft_resample(component.morph, dy, dx)
    return component


def symmetric(component, strength=1, use_prox=True, kernel=interpolation.lanczos, padding=3):
    """Make the source symmetric about its center

    See `~scarlet.operator.prox_uncentered_symmetry`
    for a description of the parameters.
    """
    step_size = component.step_morph
    center = component.pixel_center
    operator.prox_uncentered_symmetry(component.morph, step_size, center, strength, use_prox)
    return component
