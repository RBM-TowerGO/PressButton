"""Pack native RGBA atlases into RGB camera batches before host transfer.

Import only after CUDA/SimulationApp initialization. This changes byte layout,
not image values, resolution, orientation or camera order.
"""
import warp as wp


@wp.kernel
def _pack(source: wp.array(dtype=wp.uint8, ndim=3),
          destination: wp.array(dtype=wp.uint8, ndim=4),
          width: int, height: int, columns: int):
    camera, y, x = wp.tid()
    sx = (camera % columns) * width + x
    sy = (camera // columns) * height + y
    destination[camera, y, x, 0] = source[sy, sx, 0]
    destination[camera, y, x, 1] = source[sy, sx, 1]
    destination[camera, y, x, 2] = source[sy, sx, 2]


def pack_rgb(source, destination, columns):
    """Write an existing contiguous CUDA (N,H,W,3) tensor asynchronously."""
    n, height, width, channels = destination.shape
    if channels != 3 or source.ndim != 3 or source.shape[2] not in (3, 4):
        raise ValueError('Expected RGBA/RGB atlas and RGB batch')
    if tuple(source.shape[:2]) != (((n + columns - 1) // columns) * height, columns * width):
        raise ValueError('Atlas geometry does not match camera batch')
    wp.launch(_pack, dim=(n, height, width),
              inputs=[wp.from_torch(source), wp.from_torch(destination), width, height, columns],
              device=str(destination.device), stream=wp.stream_from_torch(__import__('torch').cuda.current_stream(destination.device)))
