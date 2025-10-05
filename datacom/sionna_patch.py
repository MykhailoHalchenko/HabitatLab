# Patch to disable ray tracing in Sionna
# when not needed, avoiding heavy dependencies.
import sys

class FakeRTModule:
    def __getattr__(self, name):
        raise ImportError(f"Ray tracing is disabled. '{name}' unavailable.")

sys.modules['sionna.rt'] = FakeRTModule()