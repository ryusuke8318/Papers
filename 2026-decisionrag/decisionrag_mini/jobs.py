"""実行の設定。GPU の種類は、環境変数 DR_GPU に Modal の GPU 名を入れて切り替える(既定は VRAM 80GB の GPU)。

メモリと時間の目安(Qwen3-1.7B-Base、LoRA、bf16。1 例は最大 12,000 トークン。学習 1 エポック = 問い 4,000 件 = 約 6,000 万トークン):
  VRAM   LoRA は約 12〜16GB(24GB の GPU で足りる)。全パラメータの学習(--full)は約 40GB 以上
  時間   VRAM 80GB の GPU で約 1.5 時間 / 同世代の最上位で約 0.8 時間 / VRAM 48GB の GPU で約 3 時間 / VRAM 24GB の GPU で約 8〜9 時間(FLOPs からの見積もり。実測ではない)
  評価   全実験(main、pool_scaling、request_size、choice、budget、latency)で、VRAM 80GB の GPU で約 0.5 時間
"""
MODEL = "Qwen/Qwen3-1.7B-Base"

TRAIN = dict(n_train_questions=4000, epochs=1, batch=8, max_tokens=12000, lr_lora=1e-4, lr_full=1e-5, lr_head=1e-4)
EVAL = dict(n_eval=300, exps="main,pool_scaling,request_size,choice,budget,latency")
SMOKE = dict(n_train_questions=40, limit=16, n_eval=10, exps="main,pool_scaling,choice,latency")
