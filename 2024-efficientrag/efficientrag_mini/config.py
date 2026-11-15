"""設定。論文との違いは README の対応表を参照。"""
SYNTH_LLM = "Qwen/Qwen2.5-7B-Instruct"        # 合成データの生成・最終の回答・LLM ベースの基準手法・正誤判定の LLM(論文は 70B / 8B / GPT-3.5)
ENCODER = "microsoft/deberta-v3-large"         # Labeler / Filter のエンコーダ(論文と同じ)
EMBED = "BAAI/bge-small-en-v1.5"               # 検索の埋め込みモデル(論文は contriever-msmarco)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DATASETS = ("hotpotqa", "2wiki", "musique")
N_TRAIN = {"hotpotqa": 3000, "2wiki": 1500, "musique": 1500}   # 合成データを作る質問の数
N_EVAL = 300                                   # 評価の質問数
CORPUS_QUESTIONS = 2000                        # 検索の対象とするコーパス: 評価質問を含む、開発用の 2,000 問の段落
MAX_CHUNK_WORDS = 256                          # Labeler に入れるチャンクの語数の上限

TRAIN = dict(epochs=3, lr=2e-5, batch=16, max_len=512, warmup=0.1, neg_per_query=2)
INFER = dict(k=5, max_iter=4, threshold=0.5)   # 1 回の検索で見る件数 / 最大反復数 / <Continue> の確率の閾値
