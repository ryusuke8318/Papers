# RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs

- 著者: Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, Bryan Catanzaro (Georgia Tech / NVIDIA)
- 年: 2024 (arXiv:2407.02485, NeurIPS 2024)
- タグ: RAG, reranking, instruction-tuning, データ効率, ドメイン汎化

## 一言でいうと

RAG の「rerank 用の別モデル」を捨て、1 つの LLM に ranking と生成の両方を instruction tuning するレシピの論文です。学習ブレンドに少量 (約 50k ペア) の ranking データを混ぜるだけで、10 倍量で学習した専門 ranker を上回り、9 つの一般ベンチマークと 5 つの医療ベンチマーク (学習なし) で ChatQA-1.5 や GPT-4 を上回ります。

## 背景・課題

通常の RAG (retrieve-then-generate) には三つのつらさがあります。

- LLM は top-k 件の読みには強いが、top-100 を丸ごと読ませても精度は伸びない (Fig.1: ChatQA-1.5 は k=10 付近で飽和)。小さい k は recall 不足、大きい k はノイズ混入のトレードオフです。
- 小さい k で当てるには retriever の recall が要るが、bi-encoder の独立エンコードでは表現力不足で取りこぼします。
- 別立ての ranker (BERT/T5 系) を挟む手はあるが、LLM 本体ほどのゼロショット汎化はなく、パイプラインも重くなります。

## 提案手法

二段階ファインチューニングで、単一 LLM に二役をさせます。

- Stage-I (SFT): 会話・ELI5・Self-Instruct 等の高品質 instruction データ 128K 件で素振り (batch 128, 1000 steps, lr 5e-6)。
- Stage-II (統一 instruction tuning): 次の 5 種をすべて同じ (x=質問, c=文脈, y=答え) 形式に畳んで学習します (batch 64, 3300 steps ≒ 1 epoch, lr 3e-7 (8B) / 2e-7 (70B))。
  1. Stage-I の SFT データ (維持用)
  2. Context-rich QA (DROP, NarrativeQA, Quoref, ROPES, NewsQA, TAT-QA、会話 QA)
  3. Retrieval-augmented QA (SQuAD + WebQuestions。各問に golden + BM25 検索の 5 文脈)
  4. Context ranking (MS MARCO の golden/ BM25 hard-negative に True/False、+ 会話 QA から 4-gram recall ≥0.5/<0.1 で作った疑似ペア。計約 50k)
  5. Retrieval-augmented ranking (SQuAD + WebQuestions の 5 文脈から関連 passage の id を列挙)

推論は retrieve-rerank-generate の三段です。(1) retriever が top-N 件 (8B: N=100、70B: N=30) を取る。(2) RankRAG が各文脈に「True を生成する確率」を relevance score として付け替え、top-k (k≪N、例 5) に絞る。(3) その top-k で答えを生成します。rerank は 1 トークン生成相当なので、N=20→100 で EM +5.9〜9.1% に対し時間は +0.9x〜6.0x に収まります (20〜100x にはならない)。

## 実験の主な結果

9 データセットのゼロショット比較です (Tab. 2、EM。TriviaQA/PopQA は EM/Acc、HotpotQA/2WikimQA は EM/F1)。

| 手法 | NQ | TriviaQA | PopQA | HotpotQA | 2WikimQA | FEVER | Doc2Dial | TopiOCQA | Inscit | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama3-Instruct 8B | 30.9 | 70.7 | 34.9 | 26.0 | 9.6 | 88.9 | 33.6 | 44.9 | 32.6 | 40.8 |
| Llama3-ChatQA-1.5 8B | 42.4 | 81.0 | 52.6 | 33.4 | 26.8 | 90.9 | 39.3 | 49.9 | 30.1 | 49.6 |
| Llama3-RankRAG 8B | 50.6 | 82.9 | 57.6 | 35.3 | 31.4 | 92.0 | 40.4 | 50.4 | 33.3 | 52.6 |
| Llama3-ChatQA-1.5 70B | 47.0 | 85.6 | 50.9 | 42.2 | 34.9 | 92.7 | 41.3 | 55.6 | 32.3 | 53.6 |
| Llama3-RankRAG 70B | 54.2 | 86.5 | 59.9 | 42.7 | 38.2 | 93.8 | 41.5 | 52.8 | 35.2 | 56.1 |
| GPT-4-0613 (RAG なし) | 40.3 | 84.8 | 31.3 | 34.5 | 29.8 | 87.7 | 27.6 | 30.1 | 27.0 | 42.0 |

- 8B でも InstructRetro 43B (5 倍) や RA-DIT 65B (8 倍) を上回り、Llama3-Instruct 70B を NQ/TriviaQA で抜きます。
- 伸びは難しいデータほど大きく、PopQA・2WikimQA では ChatQA-1.5 比 +10% 超です。
- GPT-4 に RAG を付けると逆に落ちることがあります (†: 関連文脈がないと回答拒否するため)。

アブレーションです (Tab. 3、Llama3-8B、Avg.)。RankRAG 8B 52.6 に対し、w/o reranking 49.8、w/o RQA 51.6、w/o RAR 51.8、w/ RAFT 48.1、Stage-I SFT のみ 42.2。rerank を抜くと全タスクで低下します。

ranking 性能です (Tab. 6、NQ の R@5。Dragon retriever 単体 74.9)。RankLLaMA 8B 77.8 (約 503k)、BGE-Reranker 78.0 (*)、GPT-4 79.3 に対し、RankRAG 8B は約 50k で 80.3。(*: BGE は評価タスクと重なる学習データを含むと論文に明記)。10 倍量の専門 ranker を少量データで上回ります。一方、ranking 学習なしの ChatQA-1.5 をそのまま ranker に使うと 68.2 と retriever 単体より悪化します。Fig.4 では ranking データ 5k 件 (MS MARCO の約 1%) でも十分強く、50k への増加は non-marginal な上積みです。

医療ベンチマーク Mirage です (Tab. 5、医療データでの追加学習なし)。RankRAG 8B Avg. 63.95 (ChatQA-1.5 8B 59.96、Meditron 70B に +6.3%)、RankRAG 70B Avg. 78.06 (ChatQA-1.5 70B 73.98、GPT-4 の 98% 超)。

## 実装メモ

LLM の学習は重いので、このリポジトリでは範囲を絞り、**主張の向きを合成データで確かめる小規模実装**を `src/rankrag_toy.py` に置きました。依存は numpy のみです。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/rankrag_toy.py
```

実装の対応関係は次のとおりです。

| 関数 | 対応する主張 |
|------|--------------|
| `exp_rerank_gain` | Tab.3: 生成時 top-k を rerank 順にする vs retriever 順のまま (w/o reranking) |
| `exp_data_efficiency` | Tab.6 + Fig.4: 少量+統一学習の scorer vs 10 倍量の ranking-only 専門 scorer vs 未学習 ranker |
| `exp_topn_tradeoff` | Fig.1 対 §5.5: k を変えたときの EM。rerank ありは小さい k で飽和 |

合成の仕掛けは `make_corpus` に集約しています。各 passage は `s_lex` (語彙っぽい信号)、`s_dense` (retriever が使う弱い bi-encoder 信号)、`s_ans` (答えの有無の oracle。統一 (x,c,y) 学習で ranking head が触れるという転移の代理、仮定) を持ちます。生成 EM の代理は「golden が top-k 内 AND top-k 内 distractor に誤らされない (1 件あたり確率 0.03、仮定)」です。

実行結果は次のとおりです。

```
[Tab.3] rerankなし EM@5: 0.132 (DEGENERATE)
[Tab.3] rerankあり EM@5: 0.658 / 改善: +0.527
[Tab.6] backbone retriever R@5: 0.146
[Tab.6] 未学習 ranker R@5: 0.044 (DEGENERATE、backbone を下回る)
[Tab.6] 専門 ranker(10x量) R@5: 0.341
[Tab.6] 統一学習(1x量) R@5: 0.732
[Fig.1/6] k= 1: rerankなし 0.037 <- DEGENERATE(小さい k で崩壊) / rerankあり 0.451
[Fig.1/6] k= 2: rerankなし 0.062 <- DEGENERATE(小さい k で崩壊) / rerankあり 0.569
[Fig.1/6] k= 5: rerankなし 0.117 / rerankあり 0.657
[Fig.1/6] k=10: rerankなし 0.166 / rerankあり 0.656
[Fig.1/6] k=20: rerankなし 0.221 / rerankあり 0.540
```

- rerank ありは +0.527 で上回り、向きは Tab.3 (52.6 対 49.8) と同じです。絶対値は比べられません。
- 少量+統一学習 (0.732) が 10 倍量の専門 scorer (0.341) を上回り、未学習 ranker (0.044) は backbone (0.146) より悪化します。向きは Tab.6 (80.3 対 77.8、68.2 < 74.9) と同じです。ただし小規模な転移信号 `s_ans` が強めのため差は誇張されています。一致と言えるのは順序だけです。
- rerank ありは k=5 で飽和し、k=20 ではノイズで少し落ちます。rerank なしは k を増やしても recall がじわ伸びするだけです。向きは Fig.1 (rerank なしの飽和) と §E.2 (rerank 後は k=5 で十分) と同じです。

### この実装がカバーしない範囲

- LLM のファインチューニングそのもの (Llama3-8B/70B) や、9+5 ベンチマークでの再現はしていません。
- 学習ブレンド比 (Appendix B) の再現や、最適な ranking データ量の探索はしていません。
- Fig.5 の時間効率カーブや、DPR/Contriever 違い (Fig.3) の再現はしていません。
- 医療ドメインへのゼロショット汎化 (Tab.5) は対象外です。

## Modal での縮小再現 (HotpotQA x Qwen2.5-0.5B)

Tab.3 (w/o reranking で低下) と Tab.6/Fig.4 (ranking データが多いほど ranking が良くなる) を小型モデルで確かめる縮小再現です。コードは `rankrag_mini/` にあります。

### 設定

- モデル: Qwen2.5-0.5B-Instruct + LoRA (r=16, lr 2e-4, 2 epoch)
- データ: HotpotQA distractor。取得元 (curtis.ml.cmu.edu) が名前解決できないため、HF Hub キャッシュから固定 subset を切り出した `data_seed.json` (train 2000 / validation 300) を同梱し、ここから train 1000 / test 150 を作ります。1 問につき gold 2 + distractor 8 の 10 文脈 (10 件未満の例は除外)
- 比較する 3 条件 (backbone retriever は語彙 overlap の弱順序。生成は top-3 文脈)
  - `qa_only`: 生成のみで学習 (ranking データ 0)
  - `rank_small`: 生成 + ranking 200 件
  - `rank_full`: 生成 + ranking 全量 (4000 件)
- rerank スコアは True/False の最終位置 logit 差です
- Modal では bf16 で学習します (fp16 では loss=nan が頻発したため。一部バッチの nan は残るが学習は継続し、最終評価は正常)

### 実行

計算の本体は `core.py` で、Modal とローカルで同じコードが動きます。`modal_app.py` は Modal につなぐ薄いラッパーです。

```bash
modal run rankrag_mini/modal_app.py --stage prepare   # データ作成 (CPU)
modal run rankrag_mini/modal_app.py --stage train     # 学習+評価 (3 条件を並列)
```

`--kinds rank_small,rank_full` のように条件を絞れます。注意: `modal run` のローカルプロセスを kill するとリモートの実行もキャンセルされます。完了までクライアントを生かしておくこと。

ローカルで動かす場合 (試運転向け。本番規模は Modal 推奨):

```bash
pip install -r rankrag_mini/requirements-local.txt
python rankrag_mini/run_local.py --smoke              # 試運転 (train 32 / test 8 / 1 epoch)
python rankrag_mini/run_local.py                      # 本番規模 (train 1000 / test 150 / 2 epoch)
```

データは `rankrag_mini/data/`、結果は `rankrag_mini/results/rankrag_*_local.json` に保存されます。

試運転の実行例: 3 条件とも学習・評価が通りました。EM はすべて 0.000 (規模が小さすぎて生成が当たらない) ですが、recall@3 は qa_only 0.312→0.750、rank_small/rank_full 0.312→0.875 と reranked が上回り、動作確認の記録としています。

### 結果 (test 150 件)

| kind | norerank EM / F1 / recall@3 | reranked EM / F1 / recall@3 |
|---|---|---|
| qa_only | 0.133 / 0.214 / 0.443 | 0.213 / 0.349 / 0.610 |
| rank_small | 0.120 / 0.217 / 0.443 | 0.167 / 0.330 / 0.817 |
| rank_full | 0.047 / 0.123 / 0.443 | 0.073 / 0.190 / 0.917 |

結果は `rankrag_mini/results/rankrag_*_mini.json` に保存してあります。

### 論文との比較

- 3 条件とも reranked EM が norerank EM を上回ります (0.213 対 0.133、0.167 対 0.120、0.073 対 0.047)。向きは Tab.3 (w/o reranking で全タスク低下) と同じです。絶対値は比べられません。
- reranked recall@3 は 0.443 (backbone) → 0.610 → 0.817 → 0.917 と単調に上がります。ranking データが多いほど ranking が良くなる向きは Tab.6/Fig.4 と同じです。
- 一致しない点: kind 間の生成 EM は qa_only が最良で、ranking 混じりは生成 EM を下げます。0.5B では生成と ranking で容量を取り合うらしく、論文の「統一学習で相互に高め合う」はこの規模では見えません。
- 「ranking 未学習モデルを ranker に使うと悪化」(Tab.6 の 68.2 < 74.9) は再現しません。backbone が語彙 overlap と弱すぎて、学習済み scorer なら何でも上回るためです。

### 気づいた点

- rank_full の生成は passage の丸写しっぽい冗長回答になりがちです (正解 Larnelle Harris に対し、来歴の長文をそのまま出す例など)。ranking 行の混入で回答形式が崩れます。
- seed 1 点のみの実行で、誤差棒はありません。

### この再現がカバーしない範囲

- Llama3-8B/70B、9+5 ベンチマーク、MS MARCO、実 retriever、医療汎化はなし。
- 別立ての専門 ranker との対決はなし。small/full は同一モデル内の ranking 量比較のみです。

## 所感

- 「ranking を QA 形式に畳むと少量データで効く」が核で、Tab.6 の 50k 対 503k が一番の売りです。
- 実務的には「rerank は 1 トークン相当で N=100 でも +0.9x〜6.0x」の計測が有用です。LLM rerank のコスト感の目安になります。

## 参考

- 論文: https://arxiv.org/abs/2407.02485
- 紹介記事 (Hugging Face paper page): https://huggingface.co/papers/2407.02485
