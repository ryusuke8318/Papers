"""条件の定義。"""
STUDENT = "Qwen/Qwen2.5-1.5B-Instruct"   # 論文は Qwen2.5-7B/32B-Instruct。縮小して 1.5B にする
TEACHER = "Qwen/Qwen3-8B"                # 教師(思考モード)。論文は DeepSeek-R1

EVAL_COMMON = dict(w=20, s=10)

JOBS: dict[str, dict] = {
    # ゼロショット
    "zs_direct": dict(kind="eval", model=STUDENT, reasoning=False, thinking=None, max_tokens=256),
    "zs_cot": dict(kind="eval", model=STUDENT, reasoning=True, thinking=None, max_tokens=2048),
    "zs_teacher": dict(kind="eval", model=TEACHER, reasoning=True, thinking=True, max_tokens=4096),
    # コールドスタート SFT(データの作り方の比較)
    "sft_direct": dict(kind="sft", data="direct", epochs=3, lr=1e-5, reasoning=False, max_tokens=256),
    "sft_reason": dict(kind="sft", data="reason", epochs=3, lr=1e-5, reasoning=True, max_tokens=3072),
    "sft_reason_nofilter": dict(kind="sft", data="nofilter", epochs=3, lr=1e-5, reasoning=True, max_tokens=3072),
    # 出力の上限を広げた評価(打ち切りが無効な出力の原因かを確かめる)。モデルは学習済みのものを使う
    "sft_reason_long": dict(kind="eval", model="/cache/rr/models/sft_reason", reasoning=True, thinking=None, max_tokens=6144),
    "sft_reason_nofilter_long": dict(kind="eval", model="/cache/rr/models/sft_reason_nofilter", reasoning=True, thinking=None, max_tokens=6144),
    # GRPO(SFT の後)。報酬の比較
    "grpo_multi": dict(kind="grpo", init="sft_reason", reward="multi", reasoning=True, max_tokens=3072),
    "grpo_ndcg": dict(kind="grpo", init="sft_reason", reward="ndcg", reasoning=True, max_tokens=3072),
    # RL のみ(SFT なし)
    "rl_only": dict(kind="grpo", init=STUDENT, reward="multi", reasoning=True, max_tokens=3072),
}

GRPO = dict(rounds=10, prompts_per_round=96, group=8, lr=1e-6, minibatch=8, max_len=7000)
SFT_MAXLEN = 7000
DATA = dict(train_pool=8000, n_train=3000, n_eval20=300, n_eval100=200, teacher_n=4, teacher_max_tokens=3072, teacher_threshold=0.9)

# 温度 0.6 のサンプリングで評価し直す(貪欲デコードでは、推論つきのモデルが同じ検討を繰り返して上限まで止まらなくなるため)。モデルは学習済みのもの
for _n in ("sft_reason", "sft_reason_nofilter", "grpo_ndcg", "grpo_multi", "rl_only"):
    JOBS[_n + "_t06"] = dict(kind="eval", model=f"/cache/rr/models/{_n}", reasoning=True, thinking=None, max_tokens=3072, temperature=0.6)
JOBS["zs_teacher_t06"] = dict(kind="eval", model=TEACHER, reasoning=True, thinking=True, max_tokens=4096, temperature=0.6)
JOBS["zs_cot_t06"] = dict(kind="eval", model=STUDENT, reasoning=True, thinking=None, max_tokens=2048, temperature=0.6)

# 質問ごとの NDCG を保存するための再評価(温度 0.6。順位のみのモデルは貪欲)
JOBS["sft_direct_pq"] = dict(kind="eval", model="/cache/rr/models/sft_direct", reasoning=False, thinking=None, max_tokens=256)
JOBS["zs_direct_pq"] = dict(kind="eval", model=STUDENT, reasoning=False, thinking=None, max_tokens=256)
# GRPO の 2 回目のシード(別の rollout の乱数と、別の質問の順番)
JOBS["grpo_multi_s1"] = dict(kind="grpo", init="sft_reason", reward="multi", reasoning=True, max_tokens=3072, seed=1, temperature=0.6)
JOBS["grpo_ndcg_s1"] = dict(kind="grpo", init="sft_reason", reward="ndcg", reasoning=True, max_tokens=3072, seed=1, temperature=0.6)

# 3B の検証(推論あり・順位のみの SFT)。教師データは 1.5B と共通(同じ Qwen2.5 のチャット形式)
STUDENT_3B = "Qwen/Qwen2.5-3B-Instruct"
JOBS["sft_reason_3b"] = dict(kind="sft", student=STUDENT_3B, data="reason", epochs=3, lr=1e-5, reasoning=True, max_tokens=3072, temperature=0.6)
JOBS["sft_direct_3b"] = dict(kind="sft", student=STUDENT_3B, data="direct", epochs=2, lr=1e-5, reasoning=False, max_tokens=256)
JOBS["sft_reason_3b_ev"] = dict(kind="eval", model="/cache/rr/models/sft_reason_3b", reasoning=True, thinking=None, max_tokens=3072, temperature=0.6)
JOBS["sft_direct_3b_ev"] = dict(kind="eval", model="/cache/rr/models/sft_direct_3b", reasoning=False, thinking=None, max_tokens=256)

# 3B の SFT(推論あり)は、課金の上限でワークスペースが止まり、3 エポックのうち 2 エポック目の終わりまでが保存された。その checkpoint を評価する
JOBS["sft_reason_3b_ep2_ev"] = dict(kind="eval", model="/cache/rr/models/sft_reason_3b_ep2", reasoning=True, thinking=None, max_tokens=3072, temperature=0.6)
