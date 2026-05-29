# Vendored: Sesame CSM

This directory is a **verbatim vendored copy** of the Sesame AI CSM reference
repository, used by Jarvis's optional CSM-1B TTS engine (`tts_engine = "csm"`).

- Upstream: https://github.com/SesameAILabs/csm
- Vendored commit: `daed31e6d42cf71873999075de204fa37d2acec3`
- Licence: see `LICENSE` in this directory (upstream Apache-2.0).

It is vendored (not pip-installed) because CSM is not published as a package.
Jarvis adds this directory to `sys.path` lazily, only when the CSM engine
initialises, so it never pollutes the global import path. See
`src/jarvis/output/tts.py::SesameCSMTTS`.

## Requirements

- A CUDA-capable GPU. RTX 50-series (Blackwell, sm_120) needs a CUDA 12.8 /
  cu128 PyTorch build.
- The CSM-1B and Llama-3.2 models are **gated** on HuggingFace. Run
  `huggingface-cli login` and accept both model licences before first use,
  otherwise `load_csm_1b()` fails and Jarvis falls back to no speech (the error
  is logged).
- Runtime deps (commented out in the repo `requirements.txt`): `moshi`,
  `torchtune`, `torchao`. `torch`/`torchaudio` already ship with Chatterbox.

## Updating

To refresh, re-clone upstream at the desired commit, copy the files here,
remove the nested `.git`, and update the commit hash above.
