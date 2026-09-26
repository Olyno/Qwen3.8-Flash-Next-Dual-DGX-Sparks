"""Import + wiring smoke test for the ported v0.30 overlay files (no GPU)."""
import inspect


def main() -> None:
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
    from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP
    from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        Qwen4ExpNGramEmbedding,
        _PlePackedTableEmbedding,
    )

    # overlay markers present
    assert "fp8dense overlay" in inspect.getsource(GatedResidual.__init__)
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForCausalLM

    src = inspect.getsource(Qwen4ExpForCausalLM.__init__)
    assert "quant_config=vllm_config.quant_config" in src, "lm_head arm"
    mtp_src = inspect.getsource(Qwen4ExpMTP.__init__)
    assert "quant_config=self.quant_config" in mtp_src, "mtp lm_head arm"

    # packed-table subclass extends the pinned-host embedding
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpPLEPinnedHostEmbedding

    assert issubclass(_PlePackedTableEmbedding, Qwen4ExpPLEPinnedHostEmbedding)
    print("IMPORT-OK all four ported files wired")


if __name__ == "__main__":
    main()
