"""Model deployment for SubCell https://github.com/afermg/SubCellPortable."""


#uv run --python .venv/bin/python python server.py tcp://0.0.0.0:5110

import sys
from functools import partial
from pathlib import Path

import numpy
import pynng
import torch
import trio
from loguru import logger
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

from process import setup_model

# We will use pre-existing information to enforce guardrails on the input data
# model -> (expected #channels, mandated shape of yx)
guardrail_shapes = {
    "mae_contrast_supcon_model": (4, 16),  # TODO relax constraints?
}

DEFAULT_ADDRESS = "tcp://127.0.0.1:5110"


def _normalize_device(device: int | str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        device = 0
    if isinstance(device, str) and device.startswith(("cuda:", "cpu")):
        return torch.device(device)
    device = int(device)
    if device < 0:
        return torch.device("cpu")
    return torch.device(f"cuda:{device}")


def setup(
    model_type: str = "mae_contrast_supcon_model",
    model_channels: str = "rybg",
    device: int | str | torch.device | None = None,
    **kwargs,
) -> dict:
    """Load a pretrained SubCell model and return a Nahual processor."""
    device = _normalize_device(device)

    execution_defaults = dict()

    setup_kwargs = kwargs.get("setup_kwargs", {})
    execution_kwargs = kwargs.get("execution_kwargs", {})

    setup_params = {
        "model_type": model_type,
        "model_channels": model_channels,
        **setup_kwargs,
    }
    execution_params = {**execution_defaults, **execution_kwargs}

    # Load model instance
    model = setup_model(**setup_params)
    model = model.to(device)
    # model.eval() gets done within setup_model

    expected_nchannels, yx_shape = guardrail_shapes[setup_params["model_type"]]
    execution_params["expected_nchannels"] = expected_nchannels
    execution_params["expected_yx"] = yx_shape

    # Generate a json-encodable dictionary to send back to the client
    info = {
        "device": str(device),
        "model_type": setup_params["model_type"],
        "model_channels": setup_params["model_channels"],
        "setup": {k: str(v) for k, v in setup_params.items()},
        "execution": {k: str(v) for k, v in execution_params.items()},
    }

    # "Freeze" model in-place
    processor = partial(process_pixels, model=model, device=device, **execution_params)
    return processor, info


async def main():
    """Main function for the asynchronous server.

    This function sets up a nng connection using pynng and starts a nursery to handle
    incoming requests asynchronously.

    Parameters
    ----------
    address : str
        The network address to listen on.

    Returns
    -------
    None
    """

    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"Pretrained SubCell server listening on {address}")
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


def process_pixels(
    pixels: numpy.ndarray,
    model,
    expected_yx: tuple[int],
    expected_nchannels: int,
    device: torch.device,
) -> numpy.ndarray:
    """Apply a pretrained model. We pass arguments that encode the necessary input shapes and number of channels to pad. We will valudate the yx dimensions and pad the channel dimension with zeros.

    Input contract (caller side)
    ----------------------------
    pixels : NCZYX float32 in [0, 1]; H, W divisible by 16; Z=1.
        Exactly 4 channels in **rybg slot order** — the channels the SubCell
        paper trained on:
            r → Microtubules
            y → ER
            b → Nucleus
            g → Protein-of-interest
        For Cell Painting that maps to [AGP, ER, DNA, Mito] (AGP stains
        actin+golgi+plasma-membrane, used as a microtubule-like proxy).
        Pass already-clipped pixels in [0, 1] — do NOT z-score; the model
        was trained on linearly-scaled raw intensities and a z-score collapses
        the embedding to near-zero cosine vs. the published output.

    Server-side normalization (applied here)
    ----------------------------------------
    None — ``model(pixels)`` runs the published forward directly.

    Output
    ------
    (N, 1536) — pool_op embedding (4 channels × 384-d each).
    """

    _, input_channels, _, *input_yx = pixels.shape

    validate_input_shape(input_yx, expected_yx)

    pixels = pad_channel_dim(pixels, expected_nchannels).copy()

    pixels_torch = torch.from_numpy(pixels).float().to(device)

    with torch.no_grad():
        embeddings = model(pixels_torch)
        # SubCell output: (N, 1536)
        embeddings_np = embeddings.pool_op.cpu().numpy()

    return embeddings_np


if __name__ == "__main__":
    address = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ADDRESS

    logger.add(Path(address.split("://", 1)[-1]).name.replace(":", "_"))

    try:
        trio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
