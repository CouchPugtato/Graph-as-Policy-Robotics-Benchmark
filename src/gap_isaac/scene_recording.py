"""Stream rendered Isaac camera frames into the native MP4 encoder."""

from pathlib import Path


class SceneRecorder:
    def __init__(self, output, world, physics_hz, fps=30):
        import numpy as np
        import omni.usd
        import video_encoding
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Sdf, UsdGeom

        from gap_isaac.camera_config import apply_pinhole_lens, build_pinhole_lens

        self.output = Path(output).resolve()
        if self.output.suffix.lower() != ".mp4" or self.output.exists():
            raise ValueError("recording requires a new MP4 path")
        if physics_hz % fps:
            raise ValueError("recording fps must divide the physics rate")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.world = world
        self.np = np
        self.fps = fps
        self.interval = physics_hz // fps
        self.steps = 0
        self.frames = 0
        self.closed = False
        stage = omni.usd.get_context().get_stage()
        source = stage.GetPrimAtPath("/World/Camera")
        if not source.IsValid():
            raise ValueError("scene camera missing")
        # keep recording optics separate from the perception camera
        destination = "/World/RecordingCamera"
        flattened = stage.Flatten()
        Sdf.CreatePrimInLayer(stage.GetSessionLayer(), "/World")
        if not Sdf.CopySpec(flattened, source.GetPath(), stage.GetSessionLayer(), destination):
            raise RuntimeError("could not create recording camera")
        apply_pinhole_lens(UsdGeom.Camera.Get(stage, destination), build_pinhole_lens())
        self.camera = Camera(destination, name="gap_recording", resolution=(640, 480), frequency=fps)
        # frame the workspace more closely without changing saved observations
        self.camera_eye = (1.35, -1.35, 1.15)
        self.camera_target = (0.35, 0.0, 0.35)
        set_camera_view(eye=np.array(self.camera_eye), target=np.array(self.camera_target),
                        camera_prim_path=destination)
        self.camera.initialize()
        for _ in range(8):
            world.render()
        rgba = self.camera.get_rgba()
        if rgba is None or rgba.shape != (480, 640, 4):
            raise RuntimeError("recording camera did not produce an RGBA frame")
        self.encoder = video_encoding.get_video_encoding_interface()
        if self.encoder is None or not self.encoder.start_encoding(str(self.output), fps, 0, False):
            raise RuntimeError("Isaac MP4 encoder could not start")
        self.encode(rgba)

    def encode(self, rgba):
        frame = self.np.ascontiguousarray(rgba, dtype=self.np.uint8)
        if frame.shape != (480, 640, 4):
            raise RuntimeError("invalid recording frame")
        if not self.encoder.encode_next_frame_from_buffer(frame, 640, 480):
            raise RuntimeError("Isaac MP4 frame encoding failed")
        self.frames += 1

    def render_due(self):
        return (self.steps + 1) % self.interval == 0

    def after_step(self):
        self.steps += 1
        if self.steps % self.interval == 0:
            self.encode(self.camera.get_rgba())

    def close(self):
        if not self.closed:
            self.encoder.finalize_encoding()
            self.closed = True
        return {"path": str(self.output), "fps": self.fps, "frames": self.frames,
                "duration_s": self.frames / self.fps, "width": 640, "height": 480,
                "encoder": "isaac-omni-videoencoding"}
