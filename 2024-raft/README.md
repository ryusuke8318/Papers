# RAFT: Adapting Language Model to Domain Specific RAG

- 著者: Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, Joseph E. Gonzalez (UC Berkeley)
- 年: 2024 (arXiv:2403.10132v2, 2024-06-05, Preprint Under Review)
- タグ: RAG, ドメイン適応, ファインチューニング, distractors, Chain-of-Thought

## 一言でいうと

ドメイン特化の「開書試験」に向けた学習レシピの論文です。検索で取ってきた文書つきの質問応答 (Q + 文書 → A) をファインチューニングし、その際に「答えの根拠にならない文書 (distractor)」を混ぜ、答えに引用つきの推論過程を含めます。PubMed・HotpotQA・Gorilla (API-Bench) で、素のファインチューニングや RAG だけより一貫して良くなります。

## 背景・課題

汎用 LLM を専門ドメイン (企業の社内文書、特定の API 文書、医学文書など) で使うとき、適応方法は大きく二つあります。

- RAG による in-context 学習: テスト時に文書を付けるだけ。学習機会 (ドメインが事前に分かっていること) を活かせない。「勉強せずに開書試験を受ける」状態です。
- 素の教師ありファインチューニング (DSF): Q → A だけ、あるいは文書の暗記だけを学ぶ。テスト時の「文書つき」という形式への備えがなく、検索の不完全さ (関係ない文書が混ざる) に弱い。「閉書試験の勉強」しかしていない状態です。

実際、DSF にテスト時 RAG を付けても精度が上がらない場合があります (下表の DSF+RAG)。文書の読み方を学習していないためです。

## 提案手法

RAFT (Retrieval Augmented Fine Tuning) の学習データは次の形です。

- P% のデータ: Q + golden 文書 (D*) + distractor (D₁, D₂, ...) → CoT つきの答え (A*)
- (1−P)% のデータ: Q + distractor のみ → A* (golden なし。暗記を強制する)

要点は三つです。

1. distractor を混ぜて学習する: テスト時は top-k 検索の結果 (当たり外れ混じり) が来るので、学習時から golden 1 件 + distractor 4 件の形で「無視すべき文書の無視し方」を学ばせます。golden だけで学習すると頑健性が落ちます (Fig. 6)。
2. P < 100% にする: 一部の学習データから golden を抜きます。検索が外したときに答えられるよう、暗記 (memorization) の経路も残すためです。最適な P はデータセットで違い (40〜100%)、P = 80% 前後が良いことが多いです (Fig. 5)。
3. Chain-of-Thought + 引用つきの答え: 答えだけでなく、根拠の引用を ##begin_quote## ... ##end_quote## で示す推論過程を付けます。答えのみの学習は loss が速く下がり過学習しやすいのに対し、推論過程が精度を上げます (Tab. 2)。

テスト時は通常の RAG と同じ (top-k 文書を付けて質問) で、retriever には依存しません。

## 実験の主な結果

LLaMA2-7B のファインチューン (RAFT-7B) とベースラインの比較です (Tab. 1)。

| 手法 | PubMed | HotPot | HuggingFace | Torch Hub | TensorFlow |
|---|---|---|---|---|---|
| GPT-3.5 + RAG | 71.60 | 41.5 | 29.08 | 60.21 | 65.59 |
| LLaMA2-7B | 56.5 | 0.54 | 0.22 | 0 | 0 |
| LLaMA2-7B + RAG | 58.8 | 0.03 | 26.43 | 08.60 | 43.06 |
| DSF | 59.7 | 6.38 | 61.06 | 84.94 | 86.56 |
| DSF + RAG | 71.6 | 4.41 | 42.59 | 82.80 | 60.29 |
| RAFT | 73.30 | 35.28 | 74.00 | 84.95 | 86.86 |

- RAFT が全ドメインで最良です。HotpotQA (+30.87 対 DSF+RAG) や HuggingFace (+31.41 対 DSF+RAG) の伸びが大きいです。
- PubMed (yes/no の二値問題) では DSF+RAG との差は小さいです。
- CoT なしとの比較 (Tab. 2): HotpotQA で +9.66、HuggingFace で +14.93 など、推論過程つきが大きく上回ります (Torch Hub のみ CoT なし 86.56 が上)。
- distractor 数の汎化 (Fig. 6): golden だけで学習したモデルは test 時の文書数変化に弱く、distractor 混じりで学習したモデルが頑健です。最適な学習時 distractor 数はドメインで違います。

## 実装メモ

LLM のファインチューニングは重いので、このリポジトリでは範囲を絞り、**主張の向きを合成データで確かめる小規模実装**を `src/raft_toy.py` に置きました。依存は numpy のみです。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/raft_toy.py
```

実装の対応関係は次のとおりです。

| 関数 | 対応する主張 |
|------|--------------|
| `train_selector` / `eval_selector` | golden と distractor の中から golden を選ぶ線形 scorer (q^T W d の softmax)。distractor 混じりの学習データを再現 |
| `exp_distractors` | Fig. 6: golden のみ学習 vs distractor 混じり学習を、test 時の文書数を変えて比較 |
| `train_memorizer` / `exp_p_ratio` | Fig. 5: golden なしサンプルだけで暗記 head を学習。P=100% と P=80% を検索失敗率 0 / 0.3 の下で比較 (reader 精度 0.90 は oracle として固定) |
| `exp_cot` | Tab. 2: 同じ 60 件で flat 20-way (答えのみ) と階層予測 (5-way グループ推論→グループ内 4-way、CoT の代理) を比較 |

実行結果は次のとおりです。

```
[Fig.6] test 文書数を変えた時の golden 選択精度:
  test distractor 0: golden-only 1.000 / RAFT 1.000
  test distractor 1: golden-only 0.522 / RAFT 0.996
  test distractor 3: golden-only 0.287 / RAFT 0.995
  test distractor 4: golden-only 0.215 / RAFT 0.990
  test distractor 9: golden-only 0.126 / RAFT 0.981
[Fig.5] P=1.0: memorizer 精度 0.050 (学習サンプル 0 件)
[Fig.5] P=0.8: memorizer 精度 0.997 (学習サンプル 100 件)
[Fig.5] 検索失敗率 0.0 での総合精度: P=100% 0.900 / P=80% 0.900
[Fig.5] 検索失敗率 0.3 での総合精度: P=100% 0.645 / P=80% 0.929
[Tab.2] 答えのみ: テスト精度 0.472 (5 seed 平均)
[Tab.2] CoTあり: テスト精度 0.487 (5 seed 平均)
[Tab.2] CoT の改善: +0.015
```

- golden のみ学習は、候補が golden 1 件だけなので loss が常に 0 で何も学ばず、test 時はほぼ chance (1/(k+1)) になります。distractor 混じりの学習は test 文書数によらず 0.98 以上を保ちます。向きは論文の Fig. 6 と同じです。
- P=80% は検索失敗がない場合は P=100% と同じ (0.900)、失敗率 0.3 では 0.929 対 0.645 で上回ります。向きは論文の Fig. 5 と同じです。
- CoT の改善は +0.015 と小さく、論文の +9〜15% とは桁が違います。線形の小規模設定では推論の効果がほぼ「分類の階層化」にしかならず、LLM の生成過程で起きる過学習防止の効果は再現できていません。一致したと言えるのは向きだけです。

### この実装がカバーしない範囲

- LLM のファインチューニングそのもの (LLaMA2-7B 等) や、PubMed・HotpotQA・Gorilla での再現はしていません。
- CoT の引用生成 (GPT-4 による推論チェーンの作成) や、distractor 数の厳密な最適化も対象外です。
- Fig. 5 の「最適な P がデータセットで違う」ことの再現はしていません (P=80% の一点のみ)。

## Modal での縮小再現 (HotpotQA x Qwen2.5-0.5B)

論文 Table 1 の順序 (RAFT > DSF+RAG ≧ DSF) を小型モデルで確かめる縮小再現です。コードは `hotpot_mini/` にあります。

### 設定

- モデル: Qwen2.5-0.5B-Instruct + LoRA (r=16, lr 2e-4, 2 epoch)。GPU は A10G
- データ: HotpotQA distractor (train 3000、test 300)。1 問につき golden + distractor 4 件
- 学習データの CoT 回答は supporting facts からテンプレートで生成 (引用つき。GPT-4 は使わない)
- 比較する 3 条件 (学習するのは dsf と raft のみ。dsf_rag は dsf の重みを RAG で評価 = 論文の DSF+RAG と同じ定義)
  - `dsf`: Q→A のみで学習、Q のみで評価
  - `dsf_rag`: dsf の重みで Q+文書で評価
  - `raft`: Q+文書→引用つき CoT で学習、Q+文書で評価
- 検索はランダム distractor で代用 (実 retriever なし)

### 実行

計算の本体は `core.py` で、Modal とローカルで同じコードが動きます。`modal_app.py` は Modal の GPU につなぐ薄いラッパーです。

**Modal で動かす場合** (複数の run を別々の GPU で並列に回せます)

```bash
modal run hotpot_mini/modal_app.py --stage prepare   # データ作成 (CPU)
modal run hotpot_mini/modal_app.py --stage train     # 学習+評価 (A10G x2 並列)
```

注意: `modal run` のローカルプロセスを kill するとリモートの実行もキャンセルされます。`--detach` を付けても実行中の入力はキャンセルされたので、完了までクライアントを生かしておくこと (フォアグラウンド実行か、kill されないバックグラウンド実行)。

**ローカルで動かす場合** (試運転向け。本番規模は Modal 推奨)

```bash
pip install -r hotpot_mini/requirements-local.txt
python hotpot_mini/run_local.py --smoke              # 試運転 (train 64 / test 16 / 1 epoch)
python hotpot_mini/run_local.py                      # 本番規模 (train 3000 / test 300 / 2 epoch)
```

データは `hotpot_mini/data/`、結果は `hotpot_mini/results/hotpot_*_local.json` に保存されます。GPU (VRAM 12GB のローカル GPU で確認) があれば試運転は数分で終わります。CPU だけでも動きますが、試運転でも時間がかかります。

試運転の実行例 (VRAM 12GB のローカル GPU、train 64 / test 16 / 1 epoch、データ取得を除く): dsf 約 5 分、raft 約 3 分。結果は dsf EM .062 / F1 .128、raft EM .062 / F1 .121、dsf_rag EM .000 / F1 .024 でした。test 16 件のため 1 件が EM .0625 分あり、順序の検証ではなく動作確認の記録です。

### 結果 (test 300 件)

| variant | EM | F1 |
|---|---|---|
| dsf | .007 | .034 |
| dsf_rag | .040 | .099 |
| raft | .150 | .242 |

結果は `hotpot_mini/results/hotpot_*_mini.json` に保存してあります。

### 論文との比較

- 順序は同じです。raft が最良で、dsf_rag は dsf より少し良いだけです。文書の読み方を学習しない DSF の重みは RAG を付けても伸びない、という論文の主張と向きが一致します。なお論文の HotPot では DSF が DSF+RAG をわずかに上回る逆転がありますが、どちらもフロア水準の差です。
- 絶対値は比べられません (0.5B 対 7B、3000 問対フル、テンプレート CoT、検索なし)。

### 気づいた点

- raft の予測は、引用つきで正解する例 (Rhodes College の文書から Presbyterian Church) と、関係文書を引用しながら答えを外す例が混ざります。読む仕組み自体は動いています。
- dsf_rag の予測は前置きの長い生成になりがちです。Q のみの学習では回答形式が定まらないためです。

### この再現がカバーしない範囲

- P% の混合なし (P=100% のみ)。PubMed・Gorilla なし。実 retriever なし。chat template なし。CoT はテンプレート生成です。

## 所感

- DSF+RAG が DSF 単体より悪化する (HuggingFace 61.06→42.59 など) のが、この論文の一番の動機づけになっています。文書の読み方を学習しないと RAG は足しにならない、というのは実務でも当たりがちです。
- P% の議論と distractor 数の議論はどちらも「学習時のノイズ量の調整」で、retriever の質やドメインの難しさとの関係は論文から読み取れません。
- CoT の効果が Torch Hub では逆転している (86.56 対 84.95) のに、本文での説明が薄いです。API 呼び出し生成のように答えの形式が厳密なタスクでは、長い推論が邪魔をするのかもしれません。

## 参考

- 論文: https://arxiv.org/abs/2403.10132