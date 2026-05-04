"""Tests for OmniVoice TTS implementation."""

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestOmniVoiceTTSInterface:
    """OmniVoiceTTS must expose the same interface as the other TTS engines."""

    def test_has_required_methods(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=False)

        for name in (
            "start", "stop", "speak", "interrupt",
            "is_speaking", "get_last_spoken_text",
            "_speak_once", "_notify_speaking_state",
        ):
            assert hasattr(tts, name), f"missing method: {name}"
            assert callable(getattr(tts, name))

    def test_initialization_disabled(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=False)

        # Must not crash when disabled
        tts.start()
        tts.speak("test text")
        assert tts.is_speaking() is False
        tts.interrupt()
        tts.stop()

    def test_initialization_with_all_parameters(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(
            enabled=True,
            voice="ignored",
            rate=200,
            device="cpu",
            ref_audio_path="/path/to/ref.wav",
            instruct="male, low pitch, british accent",
            num_step=16,
            speed=1.2,
        )

        assert tts.enabled is True
        assert tts.voice == "ignored"
        assert tts.rate == 200
        assert tts.device == "cpu"
        assert tts.ref_audio_path == "/path/to/ref.wav"
        assert tts.instruct == "male, low pitch, british accent"
        assert tts.num_step == 16
        assert tts.speed == 1.2

    def test_default_values(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS()
        assert tts.enabled is True
        assert tts.device == "cuda"
        assert tts.ref_audio_path is None
        assert tts.instruct is None
        assert tts.num_step == 32
        assert tts.speed == 1.0


class TestOmniVoiceTTSBehaviour:
    """Behavioural tests for queueing, interruption, and state."""

    def test_speak_queues_text(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=True)
        # Don't actually start the worker thread for this test
        tts.speak("Hello world")
        assert not tts._q.empty()

    def test_speak_does_nothing_when_disabled(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=False)
        tts.speak("Hello world")
        assert tts._q.empty()

    def test_speak_does_nothing_for_empty_text(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=True)
        tts.speak("")
        tts.speak("   ")
        assert tts._q.empty()

    def test_interrupt_sets_flag(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=True)
        assert not tts._should_interrupt.is_set()
        tts.interrupt()
        assert tts._should_interrupt.is_set()

    def test_is_speaking_returns_event_state(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=True)
        assert tts.is_speaking() is False

        tts._is_speaking.set()
        assert tts.is_speaking() is True

        tts._is_speaking.clear()
        assert tts.is_speaking() is False

    def test_get_last_spoken_text_returns_stored_text(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=True)
        assert tts.get_last_spoken_text() == ""

        tts._last_spoken_text = "Hello world"
        assert tts.get_last_spoken_text() == "Hello world"


class TestOmniVoiceTTSErrorHandling:
    """Error-handling around model load and synthesis."""

    def test_missing_omnivoice_dependency(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=True)

        # Force the import of `omnivoice` to fail
        with patch.dict("sys.modules", {"omnivoice": None}):
            result = tts._ensure_initialized()

        assert result is False
        assert tts._init_error is not None
        assert "omnivoice" in tts._init_error.lower()

    def test_oom_during_generation_logs_and_returns(self):
        """torch.cuda.OutOfMemoryError must be caught, logged, and not crash."""
        from src.jarvis.output.tts import OmniVoiceTTS
        import sys
        import types
        from contextlib import contextmanager

        # Build minimal fake torch / sounddevice / numpy modules so that
        # _speak_once reaches the model.generate() call and exercises the
        # OOM recovery path. We don't care about playback here.
        class _FakeOOM(Exception):
            pass

        empty_cache_calls = {"n": 0}

        def _empty_cache():
            empty_cache_calls["n"] += 1

        @contextmanager
        def _no_grad():
            yield

        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = _no_grad
        fake_cuda = types.ModuleType("torch.cuda")
        fake_cuda.OutOfMemoryError = _FakeOOM
        fake_cuda.empty_cache = _empty_cache
        fake_torch.cuda = fake_cuda

        # Stub sounddevice and numpy — not actually used because generate()
        # raises OOM before any playback path runs.
        fake_sd = types.ModuleType("sounddevice")
        fake_np = types.ModuleType("numpy")

        tts = OmniVoiceTTS(enabled=True)
        # Pretend the model is loaded to skip _ensure_initialized's heavy path
        tts._initialized = True
        tts._model = MagicMock()
        tts._model.generate.side_effect = _FakeOOM("CUDA OOM")

        stubs = {
            "torch": fake_torch,
            "torch.cuda": fake_cuda,
            "sounddevice": fake_sd,
            "numpy": fake_np,
        }
        with patch.dict(sys.modules, stubs):
            # _speak_once must not raise
            tts._speak_once("Hello world")

        # empty_cache invoked exactly once during recovery
        assert empty_cache_calls["n"] == 1
        # State left clean
        assert tts.is_speaking() is False


class TestOmniVoiceTTSFactory:
    """Tests for create_tts_engine wiring."""

    def test_creates_omnivoice_engine(self):
        from src.jarvis.output.tts import create_tts_engine, OmniVoiceTTS

        tts = create_tts_engine(engine="omnivoice", enabled=False)
        assert isinstance(tts, OmniVoiceTTS)

    def test_creates_omnivoice_engine_case_insensitive(self):
        from src.jarvis.output.tts import create_tts_engine, OmniVoiceTTS

        tts1 = create_tts_engine(engine="OMNIVOICE", enabled=False)
        tts2 = create_tts_engine(engine="OmniVoice", enabled=False)
        assert isinstance(tts1, OmniVoiceTTS)
        assert isinstance(tts2, OmniVoiceTTS)

    def test_passes_omnivoice_parameters(self):
        from src.jarvis.output.tts import create_tts_engine, OmniVoiceTTS

        tts = create_tts_engine(
            engine="omnivoice",
            enabled=True,
            voice="ignored",
            rate=200,
            omnivoice_device="cpu",
            omnivoice_ref_audio="/path/ref.wav",
            omnivoice_instruct="female british accent",
            omnivoice_num_step=8,
            omnivoice_speed=0.8,
        )
        assert isinstance(tts, OmniVoiceTTS)
        assert tts.device == "cpu"
        assert tts.ref_audio_path == "/path/ref.wav"
        assert tts.instruct == "female british accent"
        assert tts.num_step == 8
        assert tts.speed == 0.8

    def test_other_engines_unaffected(self):
        from src.jarvis.output.tts import create_tts_engine, PiperTTS, ChatterboxTTS

        assert isinstance(create_tts_engine(engine="piper", enabled=False), PiperTTS)
        assert isinstance(create_tts_engine(engine="chatterbox", enabled=False), ChatterboxTTS)


class TestOmniVoiceTTSConfig:
    """Settings dataclass and load_settings round-trip."""

    def test_config_has_omnivoice_fields(self):
        from src.jarvis.config import Settings
        import inspect

        params = set(inspect.signature(Settings).parameters.keys())
        for f in (
            "tts_omnivoice_device",
            "tts_omnivoice_ref_audio",
            "tts_omnivoice_instruct",
            "tts_omnivoice_num_step",
            "tts_omnivoice_speed",
        ):
            assert f in params, f"missing settings field: {f}"

    def test_default_config_has_omnivoice_values(self):
        from src.jarvis.config import get_default_config

        defaults = get_default_config()
        assert defaults["tts_omnivoice_device"] == "cuda"
        assert defaults["tts_omnivoice_ref_audio"] is None
        assert defaults["tts_omnivoice_instruct"] is None
        assert defaults["tts_omnivoice_num_step"] == 32
        assert defaults["tts_omnivoice_speed"] == 1.0

    def test_omnivoice_engine_preserved(self):
        from src.jarvis.config import load_settings

        config_data = {"tts_engine": "omnivoice", "_config_version": 1}
        with patch("src.jarvis.config._load_json", return_value=config_data):
            settings = load_settings()
            assert settings.tts_engine == "omnivoice"

    def test_invalid_omnivoice_device_falls_back(self):
        from src.jarvis.config import load_settings

        config_data = {"tts_omnivoice_device": "tpu", "_config_version": 1}
        with patch("src.jarvis.config._load_json", return_value=config_data):
            settings = load_settings()
            assert settings.tts_omnivoice_device == "cuda"


class TestOmniVoiceTTSThreadSafety:
    """Thread safety smoke tests."""

    def test_multiple_interrupts_safe(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=True)
        for _ in range(10):
            tts.interrupt()

    def test_start_stop_cycle(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=False)
        for _ in range(3):
            tts.start()
            tts.stop()

    def test_concurrent_speaks(self):
        from src.jarvis.output.tts import OmniVoiceTTS

        tts = OmniVoiceTTS(enabled=True)

        def speak_text():
            for _ in range(10):
                tts.speak("Hello world")

        threads = [threading.Thread(target=speak_text) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
