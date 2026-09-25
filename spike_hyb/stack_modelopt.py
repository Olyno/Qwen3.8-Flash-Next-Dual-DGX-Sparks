#!/usr/bin/env python3
"""Apply the fp8dense overlay semantics on top of serve/files/modelopt_patched.py.

serve/files/modelopt_patched.py = image modelopt.py + FP8_BLOCK_SCALES MoE support
(the TP1 PLE-offload recipe needs it). The fp8dense overlay diffs were authored
against the bare image file, so we stack them manually:

1. per-channel FP8 sibling configs (fp8_pcpt_config / fp8_pbwo_config)
2. dispatch arms: quant_algo FP8_PER_CHANNEL_PER_TOKEN / FP8_PB_WO
3. prefix candidates for the text-only entry point (model.* -> model.language_model.*)
   and a bare "lm_head" key.

Refuses to patch if any anchor is missing (drift guard).
"""
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/work/modelopt_patched.py"
DST = sys.argv[2] if len(sys.argv) > 2 else "/work/modelopt_hybrid.py"

s = open(SRC).read()
orig = s

# 1+2a. sibling configs right before `return cls(` in the MIXED_PRECISION from_config
anchor1 = """        return cls(
            kv_cache_quant_method=kv_cache_quant_method,"""
assert s.count(anchor1) == 1, f"anchor1 count {s.count(anchor1)}"
repl1 = """        cfg = cls(
            kv_cache_quant_method=kv_cache_quant_method,"""
s = s.replace(anchor1, repl1)
anchor2 = """            mxfp8_config=mxfp8_config,
        )"""
assert s.count(anchor2) == 1, f"anchor2 count {s.count(anchor2)}"
s = s.replace(anchor2, anchor2 + """
        # [fp8dense overlay] sibling configs for per-channel FP8 dense layers
        # (dynamic per-token activations, no calibration scales needed).
        cfg.fp8_pcpt_config = ModelOptFp8Config(
            quant_method="FP8_PER_CHANNEL_PER_TOKEN",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=[],
        )
        cfg.fp8_pbwo_config = ModelOptFp8Config(
            quant_method="FP8_PB_WO",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=[],
        )
        return cfg""", 1)

# 2b. dispatch arms in get_quant_method of the MIXED_PRECISION class.
anchor3 = """            if quant_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)"""
assert s.count(anchor3) == 1, f"anchor3 count {s.count(anchor3)}"
s = s.replace(anchor3, anchor3 + """
            # [fp8dense overlay]
            if quant_algo == "FP8_PER_CHANNEL_PER_TOKEN":
                return ModelOptFp8PcPtLinearMethod(self.fp8_pcpt_config)
            if quant_algo == "FP8_PB_WO":
                return ModelOptFp8PbWoLinearMethod(self.fp8_pbwo_config)""", 1)

# 3. prefix candidates: text-only entry + bare lm_head
anchor4 = """        elif prefix.startswith("model.language_model."):
            candidates.append(
                "language_model.model." + prefix[len("model.language_model.") :]
            )

        return tuple(dict.fromkeys(candidates))"""
assert s.count(anchor4) == 1, f"anchor4 count {s.count(anchor4)}"
s = s.replace(anchor4, """        elif prefix.startswith("model.language_model."):
            candidates.append(
                "language_model.model." + prefix[len("model.language_model.") :]
            )
        elif prefix.startswith("model."):
            # [fp8dense overlay] text-only entry point strips
            # "model.language_model." to "model."; keep the HF spelling too.
            candidates.append("model.language_model." + prefix[len("model.") :])
        if prefix.startswith("language_model.lm_head") or prefix == "lm_head":
            candidates.append("lm_head")

        return tuple(dict.fromkeys(candidates))""", 1)

assert s != orig
open(DST, "w").write(s)
print(f"stacked {SRC} -> {DST} (+{len(s)-len(orig)} bytes)")
