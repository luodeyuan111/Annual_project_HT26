"""Drone_Interface package initializer.

Prefer the AirSim implementation if available; otherwise fall back
to the local/mock implementation. Exports the main helper classes
for convenient import by callers.
"""

DroneController = None
try:
    # Prefer AirSim implementation when present
    from .Drone_Interface_AirSim import DroneController  # type: ignore
except Exception:
    try:
        from .Drone_Interface import DroneController  # type: ignore
    except Exception:
        DroneController = None

# Export helpers from rgb_data_extractor if available
try:
    from .rgb_data_extractor import RGBDataExtractor, FrameBuffer  # type: ignore
except Exception:
    RGBDataExtractor = None
    FrameBuffer = None

__all__ = ['DroneController', 'RGBDataExtractor', 'FrameBuffer']
