"""共通: パス、メモリ上限、モデルの読み込み(fp16、LoRA)。bf16 は使わない。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent / "src"))

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
VRAM_CAP_GB = 15.0  # VRAM 16GB のカードに収まることを確かめるための上限


def cap_memory(gb: float = VRAM_CAP_GB) -> float:
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, gb * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    return frac


def gb(x: float) -> float:
    return round(x / 2**30, 3)


class Peak:
    """フェーズごとの torch.cuda.max_memory_allocated を記録する。"""

    def __init__(self):
        self.max: dict[str, float] = {}

    def start(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def stop(self, phase: str) -> float:
        import torch
        torch.cuda.synchronize()
        v = gb(torch.cuda.max_memory_allocated())
        self.max[phase] = max(self.max.get(phase, 0.0), v)
        return v


def load_model(adapter: str | None = None, trainable: bool = False, rank: int = 16):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    if trainable:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # LoRA の重みは fp32、土台は fp16
    return model, tok


def log_line(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def now() -> float:
    return time.time()
