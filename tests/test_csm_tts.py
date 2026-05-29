"""Tests for the Sesame CSM-1B TTS engine.

These verify behaviours of the SesameCSMTTS engine and its wiring into the
config/factory, with the heavy CSM model fully mocked so they run anywhere.
"""

import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest


class _FakeAudio:
    """Minimal stand-in for the 1-D audio tensor CSM returns.

    Tracks unsqueeze/cpu so tests can assert the saved tensor was shaped
    [1, N] and moved to CPU, as the Sesame docs require.
    """

    def __init__(self, n: int, dims: int = 1, cpu_called: bool = False):
        self._n = n
        self._dims = dims
        self.cpu_called = cpu_called

    @property
    def shape(self):
        return (1, self._n) if self._dims == 2 else (self._n,)

    def unsqueeze(self, dim):
        return _FakeAudio(self._n, dims=2, cpu_called=self.cpu_called)

    def cpu(self):
        return _FakeAudio(self._n, dims=self._dims, cpu_called=True)


def _make_loaded_engine(sample_rate: int = 24000, n_samples: int = 48000):
    """Build a SesameCSMTTS with a fake, already-loaded generator."""
    from src.jarvis.output.tts import SesameCSMTTS

    tts = SesameCSMTTS(enabled=True)
    gen = MagicMock()
    gen.sample_rate = sample_rate
    gen.generate.return_value = _FakeAudio(n_samples)
    tts._generator = gen
    tts._initialized = True
    return tts, gen


class TestSesameCSMTTSInterface:
    """Must expose the same interface as the other TTS engines."""

    def test_has_required_methods(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=False)
        for name in (
            "start", "stop", "speak", "interrupt",
            "is_speaking", "get_last_spoken_text",
            "_speak_once", "_notify_speaking_state",
        ):
            assert hasattr(tts, name), f"missing method: {name}"
            assert callable(getattr(tts, name))

    def test_initialization_disabled(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=False)
        tts.start()
        tts.speak("test text")
        assert tts.is_speaking() is False
        tts.interrupt()
        tts.stop()

    def test_initialization_with_all_parameters(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(
            enabled=True,
            voice="ignored",
            rate=200,
            device="cpu",
            speaker=3,
            max_audio_length_ms=15000,
        )
        assert tts.enabled is True
        assert tts.device == "cpu"
        assert tts.speaker == 3
        assert tts.max_audio_length_ms == 15000

    def test_default_values(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS()
        assert tts.enabled is True
        assert tts.device == "cuda"
        assert tts.speaker == 0
        assert tts.max_audio_length_ms == 30000


class TestSesameCSMTTSBehaviour:
    def test_speak_queues_text(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=True)
        tts.speak("Hello world")
        assert not tts._q.empty()

    def test_speak_does_nothing_when_disabled(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=False)
        tts.speak("Hello world")
        assert tts._q.empty()

    def test_speak_does_nothing_for_empty_text(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=True)
        tts.speak("")
        tts.speak("   ")
        assert tts._q.empty()

    def test_interrupt_sets_flag(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=True)
        assert not tts._should_interrupt.is_set()
        tts.interrupt()
        assert tts._should_interrupt.is_set()

    def test_is_speaking_returns_event_state(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=True)
        assert tts.is_speaking() is False
        tts._is_speaking.set()
        assert tts.is_speaking() is True
        tts._is_speaking.clear()
        assert tts.is_speaking() is False

    def test_get_last_spoken_text(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=True)
        assert tts.get_last_spoken_text() == ""
        tts._last_spoken_text = "Hello world"
        assert tts.get_last_spoken_text() == "Hello world"


class TestSesameCSMTTSSynthesis:
    """Behaviour of _speak_once with a mocked generator and playback stack."""

    def _patched_modules(self, busy_values):
        """Return a dict of stub modules for sys.modules patching.

        busy_values: list of return values for pygame.mixer.music.get_busy().
        """
        fake_pygame = MagicMock()
        fake_pygame.mixer.music.get_busy.side_effect = list(busy_values)
        fake_torchaudio = MagicMock()
        return {"pygame": fake_pygame, "torchaudio": fake_torchaudio}, fake_pygame, fake_torchaudio

    def test_generate_called_with_sesame_arguments(self):
        tts, gen = _make_loaded_engine()
        stubs, _pg, _ta = self._patched_modules([False])
        with patch.dict(sys.modules, stubs):
            tts._speak_once("Hello world")
        gen.generate.assert_called_once()
        _, kwargs = gen.generate.call_args
        assert kwargs["text"] == "Hello world"
        assert kwargs["speaker"] == 0
        assert kwargs["context"] == []
        assert kwargs["max_audio_length_ms"] == 30000

    def test_duration_callback_receives_samples_over_sample_rate(self):
        tts, _gen = _make_loaded_engine(sample_rate=24000, n_samples=48000)
        got = {}
        tts._duration_callback = lambda d: got.update(duration=d)
        stubs, _pg, _ta = self._patched_modules([False])
        with patch.dict(sys.modules, stubs):
            tts._speak_once("Hello world")
        assert got["duration"] == pytest.approx(2.0)

    def test_saved_tensor_is_2d_and_on_cpu(self):
        tts, _gen = _make_loaded_engine()
        stubs, _pg, fake_ta = self._patched_modules([False])
        with patch.dict(sys.modules, stubs):
            tts._speak_once("Hello world")
        fake_ta.save.assert_called_once()
        args, _kwargs = fake_ta.save.call_args
        saved_tensor = args[1]
        sample_rate = args[2]
        assert len(saved_tensor.shape) == 2  # unsqueeze(0) applied
        assert saved_tensor.cpu_called is True  # .cpu() applied
        assert sample_rate == 24000  # generator.sample_rate

    def test_completion_callback_fires_on_natural_completion(self):
        tts, _gen = _make_loaded_engine()
        fired = {"n": 0}
        tts._completion_callback = lambda: fired.__setitem__("n", fired["n"] + 1)
        stubs, _pg, _ta = self._patched_modules([False])
        with patch.dict(sys.modules, stubs):
            tts._speak_once("Hello world")
        assert fired["n"] == 1
        assert tts.is_speaking() is False

    def test_completion_callback_suppressed_on_interrupt(self):
        tts, _gen = _make_loaded_engine()
        fired = {"n": 0}
        tts._completion_callback = lambda: fired.__setitem__("n", fired["n"] + 1)
        # Playback stays "busy"; a barge-in arrives mid-playback (during wait).
        stubs, fake_pg, _ta = self._patched_modules([True, True])

        def _interrupt_during_wait(_ms):
            tts._should_interrupt.set()

        fake_pg.time.wait.side_effect = _interrupt_during_wait
        with patch.dict(sys.modules, stubs):
            tts._speak_once("Hello world")
        assert fired["n"] == 0
        fake_pg.mixer.music.stop.assert_called_once()
        assert tts.is_speaking() is False

    def test_missing_generator_skips_synthesis(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=True)
        tts._initialized = True  # skip heavy init
        tts._generator = None
        tts._model_error = "boom"
        with pytest.warns(UserWarning):
            tts._speak_once("Hello world")
        assert tts.is_speaking() is False


class TestSesameCSMTTSContext:
    """Conversational context (prior turns) wiring for prosody continuity."""

    def _engine(self, context_turns):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=True, context_turns=context_turns)
        gen = MagicMock()
        gen.sample_rate = 24000
        counter = {"n": 0}

        def _gen(**kwargs):
            counter["n"] += 1
            a = _FakeAudio(24000)
            a.tag = counter["n"]
            return a

        gen.generate.side_effect = _gen
        tts._generator = gen
        tts._initialized = True
        return tts, gen

    def _stubs(self):
        fake_pygame = MagicMock()
        fake_pygame.mixer.music.get_busy.return_value = False
        return {"pygame": fake_pygame, "torchaudio": MagicMock()}

    def test_first_utterance_has_empty_context(self):
        tts, gen = self._engine(3)
        with patch.dict(sys.modules, self._stubs()):
            tts._speak_once("First")
        assert gen.generate.call_args_list[0].kwargs["context"] == []

    def test_second_utterance_includes_prior_turn(self):
        tts, gen = self._engine(3)
        with patch.dict(sys.modules, self._stubs()):
            tts._speak_once("First")
            tts._speak_once("Second")
        ctx = gen.generate.call_args_list[1].kwargs["context"]
        assert len(ctx) == 1
        assert ctx[0].text == "First"
        assert ctx[0].speaker == tts.speaker
        assert ctx[0].audio.tag == 1  # the first turn's generated audio

    def test_context_capped_at_context_turns(self):
        tts, gen = self._engine(1)
        with patch.dict(sys.modules, self._stubs()):
            tts._speak_once("One")
            tts._speak_once("Two")
            tts._speak_once("Three")
        ctx = gen.generate.call_args_list[2].kwargs["context"]
        assert len(ctx) == 1
        assert ctx[0].text == "Two"  # only the single most recent prior turn

    def test_context_disabled_when_turns_zero(self):
        tts, gen = self._engine(0)
        with patch.dict(sys.modules, self._stubs()):
            tts._speak_once("One")
            tts._speak_once("Two")
        for call in gen.generate.call_args_list:
            assert call.kwargs["context"] == []

    def test_context_overflow_falls_back_to_empty(self):
        from src.jarvis.output.tts import SesameCSMTTS, _CSMSegment

        tts = SesameCSMTTS(enabled=True, context_turns=3)
        # Pre-seed history so the first generate() attempt carries context.
        tts._context.append(_CSMSegment(speaker=0, text="prev", audio=_FakeAudio(10)))
        gen = MagicMock()
        gen.sample_rate = 24000
        seen = []

        def _gen(**kwargs):
            seen.append(list(kwargs["context"]))
            if len(seen) == 1:
                raise ValueError("Inputs too long, must be below max_seq_len - max_generation_len: 123")
            return _FakeAudio(24000)

        gen.generate.side_effect = _gen
        tts._generator = gen
        tts._initialized = True
        with patch.dict(sys.modules, self._stubs()):
            tts._speak_once("Hello")
        assert len(seen) == 2          # retried after overflow
        assert seen[0] != []           # first attempt carried context
        assert seen[1] == []           # retry dropped context
        assert tts.is_speaking() is False


class TestSesameCSMTTSFactory:
    def test_creates_csm_engine(self):
        from src.jarvis.output.tts import create_tts_engine, SesameCSMTTS

        tts = create_tts_engine(engine="csm", enabled=False)
        assert isinstance(tts, SesameCSMTTS)

    def test_creates_csm_engine_case_insensitive(self):
        from src.jarvis.output.tts import create_tts_engine, SesameCSMTTS

        assert isinstance(create_tts_engine(engine="CSM", enabled=False), SesameCSMTTS)
        assert isinstance(create_tts_engine(engine="Csm", enabled=False), SesameCSMTTS)

    def test_passes_csm_parameters(self):
        from src.jarvis.output.tts import create_tts_engine, SesameCSMTTS

        tts = create_tts_engine(
            engine="csm",
            enabled=True,
            csm_device="cpu",
            csm_speaker=2,
            csm_max_audio_length_ms=20000,
            csm_context_turns=5,
        )
        assert isinstance(tts, SesameCSMTTS)
        assert tts.device == "cpu"
        assert tts.speaker == 2
        assert tts.max_audio_length_ms == 20000
        assert tts.context_turns == 5

    def test_other_engines_unaffected(self):
        from src.jarvis.output.tts import create_tts_engine, PiperTTS, ChatterboxTTS

        assert isinstance(create_tts_engine(engine="piper", enabled=False), PiperTTS)
        assert isinstance(create_tts_engine(engine="chatterbox", enabled=False), ChatterboxTTS)


class TestSesameCSMTTSConfig:
    def test_config_has_csm_fields(self):
        from src.jarvis.config import Settings
        import inspect

        params = set(inspect.signature(Settings).parameters.keys())
        for f in ("tts_csm_device", "tts_csm_speaker", "tts_csm_max_audio_length_ms",
                  "tts_csm_context_turns"):
            assert f in params, f"missing settings field: {f}"

    def test_default_config_has_csm_values(self):
        from src.jarvis.config import get_default_config

        defaults = get_default_config()
        assert defaults["tts_csm_device"] == "cuda"
        assert defaults["tts_csm_speaker"] == 0
        assert defaults["tts_csm_max_audio_length_ms"] == 30000
        assert defaults["tts_csm_context_turns"] == 3

    def test_csm_engine_preserved(self):
        from src.jarvis.config import load_settings

        config_data = {"tts_engine": "csm", "_config_version": 1}
        with patch("src.jarvis.config._load_json", return_value=config_data):
            settings = load_settings()
            assert settings.tts_engine == "csm"

    def test_invalid_csm_device_falls_back(self):
        from src.jarvis.config import load_settings

        config_data = {"tts_csm_device": "tpu", "_config_version": 1}
        with patch("src.jarvis.config._load_json", return_value=config_data):
            settings = load_settings()
            assert settings.tts_csm_device == "cuda"

    def test_invalid_csm_numbers_fall_back(self):
        from src.jarvis.config import load_settings

        config_data = {
            "tts_csm_speaker": "abc",
            "tts_csm_max_audio_length_ms": "xyz",
            "tts_csm_context_turns": "nope",
            "_config_version": 1,
        }
        with patch("src.jarvis.config._load_json", return_value=config_data):
            settings = load_settings()
            assert settings.tts_csm_speaker == 0
            assert settings.tts_csm_max_audio_length_ms == 30000
            assert settings.tts_csm_context_turns == 3

    def test_negative_context_turns_clamped_to_zero(self):
        from src.jarvis.config import load_settings

        config_data = {"tts_csm_context_turns": -5, "_config_version": 1}
        with patch("src.jarvis.config._load_json", return_value=config_data):
            settings = load_settings()
            assert settings.tts_csm_context_turns == 0


class TestSesameCSMTTSThreadSafety:
    def test_multiple_interrupts_safe(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=True)
        for _ in range(10):
            tts.interrupt()

    def test_start_stop_cycle(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=False)
        for _ in range(3):
            tts.start()
            tts.stop()

    def test_concurrent_speaks(self):
        from src.jarvis.output.tts import SesameCSMTTS

        tts = SesameCSMTTS(enabled=True)

        def speak_text():
            for _ in range(10):
                tts.speak("Hello world")

        threads = [threading.Thread(target=speak_text) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
