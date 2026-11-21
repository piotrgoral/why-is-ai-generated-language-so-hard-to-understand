"""NLLB-200 on Modal for the round-trip stage: the same NLLBTranslator code as the local path (fp16, length-sorted
batches, max_length 256), served on autoscaling GPUs so many sentence chunks and several experiments translate at once.

    modal deploy modal_nllb.py     # deploy (once, and after changing NLLBTranslator)
    modal run modal_nllb.py        # smoke test: EN -> JA -> EN on two sentences

Clients use translator "modal:facebook/nllb-200-3.3B" (repro/roundtrip.py: ModalNLLBTranslator)."""

import modal

APP, CLS = "nllb", "NLLB"
MODEL = "facebook/nllb-200-3.3B"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.15.0", "transformers==5.23.0", "sentencepiece==0.2.3", "syntok==1.4.4"
    )
    .env({"HF_HOME": "/cache", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_python_source(
        "repro"
    )  # NLLBTranslator itself, so outputs match the local path
)
cache = modal.Volume.from_name(
    "nllb-hf-cache", create_if_missing=True
)  # weights survive cold starts
app = modal.App(APP, image=image)


# gpu: first free type wins; L4 alone was often unavailable. Outputs can differ slightly by GPU type (~2-3% of round trips).
@app.cls(
    gpu=["L4", "A10G", "L40S"],
    volumes={"/cache": cache},
    scaledown_window=120,
    max_containers=10,
    timeout=1800,
)  # 10 = workspace GPU limit
class NLLB:
    name: str = modal.parameter(default=MODEL)

    @modal.enter()
    def load(self):
        from repro.roundtrip import NLLBTranslator

        self.tr = NLLBTranslator(self.name, batch_size=64)
        self.tr.load()  # download weights now, not on the first request
        cache.commit()  # persist the first download

    @modal.method()
    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        return self.tr(texts, src, tgt)


@app.local_entrypoint()
def smoke():
    nllb = NLLB()
    en = [
        "The cat sat on the mat.",
        "Round-trip translation paraphrases a sentence through another language.",
    ]
    ja = nllb.translate.remote(en, "eng_Latn", "jpn_Jpan")
    for e, j, r in zip(en, ja, nllb.translate.remote(ja, "jpn_Jpan", "eng_Latn")):
        print(f"{e}\n  -> {j}\n  -> {r}")
