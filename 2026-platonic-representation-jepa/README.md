# Proving and Improving the Platonic Representation Hypothesis with JEPAs

- 著者: Randall Balestriero (Brown Univ. / AMI Labs), Yann LeCun (NYU / AMI Labs)
- 年: 2026(arXiv:2610.39302v1, 2026-10-29)
- タグ: JEPA, Platonic Representation Hypothesis, 表現学習, マルチモーダル, スペクトル理論

## 一言でいうと

Platonic Representation Hypothesis (PRH) を、理想化した設定のもとで定理として証明した論文です。JEPA の positive pair が「同じシーンの二つの観測(good view)」であれば、別々のモダリティで独立に学習したエンコーダは回転を除いて同じ表現になります。

## 背景・課題

PRH は、異なるモダリティのモデルがスケールとともに同じ表現へ収束するという仮説です (Huh et al.)。これまでは観察にとどまっていて、いつ成り立つかを予測できず、整合を高める方法も示せませんでした。実際には、モデルとデータのスケールを上げることが唯一の手段になっていました。

## 理論の要点

世界を有限の状態集合 Z とし、モダリティ m は単射なレンダリング x = g_m(z) で観測します。JEPA は次の目的関数を最小化します。

L(f) = E_{(z,z')~π} l(f(g(z)), f(g(z'))) + Ω(f)

l は予測項、Ω は anti-collapse 項です。どちらも回転 O(k) に対して不変です。

| 結果                                         | 内容                                                                                                                                                                                                             |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prop. 1                                      | l = ‖a−b‖² で埋め込みを白色化すると、解は π が作るグラフのスペクトル埋め込み Φ_π(z) = (v_1(z),…,v_k(z))/√p(z) になる。最小損失は 2Σ(1−μ_i)。固有値に gap (μ_k > μ_{k+1}) があれば回転を除いて一意                |
| **Thm. 1** (Platonic Representation Theorem) | M 個のモダリティが同じ positive 分布 π を共有し、最小解が回転を除いて一意なら、すべてのエンコーダは f_m∘g_m = Q_m h* を満たす。つまり回転を除いて一致する                                                        |
| Thm. 2                                       | positive のグラフが違う場合、不一致はカーネル差 ‖K_π1 − K_π2‖ を固有値 gap δ で割った量で上から抑えられる                                                                                                        |
| Prop. 2                                      | 回転そのものは、非ペアデータの二次統計量までは決まらない。一般の位置にある k 組のペアがあれば、直交 Procrustes で決まる                                                                                          |
| Thm. 3, Prop. 3                              | 観測ごとのニュイサンス(言い回し、撮影視点など)を、自分のモダリティの方向に埋め込んでしまうと、その分は整合に使えない。good view はニュイサンスを使わず、bad view(同じ観測の破壊、たとえば crop や masking)は使う |
| Thm. 4                                       | views とデータの両方が、学習される「世界カーネル」を決める。good view は世界自身のカーネルを設計によって与える                                                                                                   |
| Prop. 4                                      | Next Token Prediction は、prefix と続きを positive とし、softmax の log-partition を anti-collapse 項とする bad-view JEPA の一種。同定は GL(k) までで、回転までは決まらない                                      |

Next Token Prediction が視覚と整合するのは主にスケールによってである、というのが論文の読みです。

## 実験の主な結果

- 同じコーパスで同じ整合に達するのに、good view は Next Token Prediction の約 1/42 のサンプルで済みます。
- 109M パラメータの text JEPA が、DINOv2 との整合で Qwen2.5-72B などの LLM を上回ります (WIT-1024, mutual 10-NN: .184 対 .154)。
- 整合度は「good view の割合」と「見たサンプル数」のスケーリング則に従います。85 件の未学習の run を、seed の揺れと同程度の精度で予測しています。
- 応用として、回転 1 つで COCO キャプショニング (CIDEr .838) を実現します。ペアなしの設定でも、既存手法の zero-shot 精度を上回っています。

## 実装メモ

論文の大規模実験(BERT と ViT を LeJEPA の目的関数で学習し、stable-pretraining を使う)は GPU が必要です。このリポジトリでは範囲を絞り、**理論の主張を有限の世界で数値的に確かめる小規模実装**を `src/platonic_theory.py` に置きました。依存は numpy のみです。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/platonic_theory.py
```

実装の対応関係は次のとおりです。

| 関数                                      | 対応する主張                                                           |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| `make_kernel`                             | 周辺分布 p を持つ対称な positive 分布 π(Sinkhorn で周辺分布を合わせる) |
| `spectral_embedding`                      | Prop. 1 の Φ_π                                                         |
| `train_jepa`                              | 白色化を罰則で課した JEPA の勾配降下(エンコーダは埋め込み表)           |
| `procrustes` / `rot_error`                | 最適回転と、回転後の誤差                                               |
| `exp_prop1_thm1`, `exp_prop2`, `exp_thm2` | Prop. 1, Thm. 1, Prop. 2, Thm. 2 の確認                                |

実行結果(seed 0, |Z|=30, k=3)は次のとおりです。

```
[Prop.1] 理論上の最小値 0.249200  Phi での予測損失 0.249200
[Prop.1] 勾配法で学習した埋め込みと Phi の回転後誤差: 1.83e-03
[Thm.1] モダリティ 0 と 1 の回転後誤差: 2.30e-04
[Thm.1] 回転を合わせる前の誤差(参考): 2.330
[Prop.2] 3 組のペアで推定した回転と全ペアの回転の差 (Frobenius): 1.20e-04
[Thm.2] 200 個の世界すべてで 誤差 <= 上界。誤差/上界 の最大値 = 0.32
```

- 別々の乱数初期化から学習した埋め込みは、回転を合わせる前は大きく食い違います(2.33)。回転を合わせると 1e-4 程度まで一致します。
- 残る 1e-3 から 1e-4 の誤差は、白色化を厳密な制約ではなく罰則 (λ=20) で課していることに由来します。
- Thm. 2 の上界は、ランダムに生成した 200 個の世界のすべてで成り立ちました。誤差は上界の 32% 以下です。

### この実装がカバーしない範囲

- bad view や Next Token Prediction のニュイサンスに関する主張 (Thm. 3, Prop. 3, 4) は、単射なレンダリングでは表現できないため実装していません。
- 論文の画像・テキストでの実験(スケーリング則、キャプショニング、ペアなし整合)も再現していません。

## Flickr30k での比較実験(Table 1 の縮小版)

論文 Sec. 3 の比較を、Modal の GPU 1枚で再現できる範囲に縮小して実装しました。コードは `flickr_table1/` にあります。

### 設定

- データ: Flickr30k(Karpathy 分割。train 29,000、val 1,014、test 1,000。1 画像に 5 キャプション)
- 画像側: 凍結した DINOv2 ViT-B/14 の CLS 特徴量(768 次元)。学習しません。論文は画像側も ViT-S で JEPA 学習しているので、ここが最大の違いです
- text 側: BERT-S 相当(幅 384、12 層、約 38M パラメータ)を最初から学習します。語彙の埋め込みを含むため、論文の 33M より大きくなっています
- JEPA の目的関数: LeJEPA(8 ビューの平均への近さ + SIGReg、λ=0.02)。バッチ 256 画像、AdamW(lr 5e-4、weight decay 0.05)、200 epoch、勾配クリッピング 1.0
- 比較する 5 つの variant
  - `good`: 8 ビューを同じ画像の別々のキャプションから作る(5 つ全てを使い、余りは重複)
  - `bad`: 8 ビューを 1 つのキャプションから作る
  - `ntp`: 因果マスクをかけた同じ BERT による Next Token Prediction(1 画像につき 1 キャプション)
  - `ntp_pairs`: 同じ画像の 2 つのキャプションを連結した系列での Next Token Prediction
  - `random`: 未学習
- 評価: 各画像の 1 番目のキャプションのバックボーン特徴量(トークンの平均)を使います
  - mutual 10-NN: 何も当てはめずに、近傍集合の重なりを測る(偶然の水準は .01)
  - R@1 / R@5: 白色化したうえで、検証ペアで 1 つの直交行を持つ写像を当てはめ、テスト 1,000 組で画像→キャプションを検索する(偶然の水準は R@1 .001)

論文に記載のない次の設定は、こちらで仮定しました。trim の長さ(連続区間の長さを [0.5n, n] から一様に選ぶ)、15% の mask、平均プーリング、3 層 MLP の projector(BatchNorm 付き)、5% のウォームアップとコサイン減衰、1 キャプションあたり最大 32 トークン。`jepa_text.py` に「仮定」とコメントしてあります。

### 実行

計算の本体は `flickr_table1/` の `extract_features.py`(特徴量の抽出)、`run.py`(学習と評価)、`baselines.py`(Qwen2.5 と未学習モデルの診断)で、ローカルでも Modal でも同じコードが動きます。`modal_app.py` は、それらを Modal の GPU に並列でつなぐだけのラッパーです。

**ローカルで動かす場合**(NVIDIA GPU が必要です。重い run は Modal のほうが速いので、試運転や小さな run 向けです)

```bash
pip install -r flickr_table1/requirements.txt
python flickr_table1/test_metrics.py                                         # 評価指標のテスト(GPU 不要)

# 試運転(数分): COCO から 512 枚だけ抽出し、短く学習する
python flickr_table1/extract_features.py --dataset coco --limit 512
python flickr_table1/run.py --dataset coco_limit512 --variants good ntp --samples 2560 --n-evals 2

# 本番の特徴量(1 回だけ)と、COCO の小さな run
python flickr_table1/extract_features.py --dataset coco
python flickr_table1/run.py --dataset coco --variants good ntp --samples 50000 100000 300000 --seeds 0 1 2

# Flickr30k (Table 1 の縮小版) と、診断
python flickr_table1/extract_features.py --dataset flickr
python flickr_table1/run.py --dataset flickr --variants random good bad ntp ntp_pairs --epochs 200
python flickr_table1/baselines.py diagnose
```

特徴量のキャッシュは `flickr_table1/data/` に置かれます(環境変数 `PLATONIC_DATA` で変更できます)。VRAM が 24GB 未満の GPU(12GB のローカル GPU など)では、`run.py` に `--grad-checkpoint` を付けてください。付けないと、バッチ 256 の学習でメモリ不足になります。`baselines.py lm` の既定は 0.5B〜3B のモデルで、7B(bf16 で約 15GB)は VRAM 16GB 未満の GPU には載りません。

**Modal で動かす場合**(複数の run を別々の GPU で並列に回せます)

```bash
pip install modal && modal setup
modal run flickr_table1/modal_app.py --stage features                       # Flickr30k の DINOv2 特徴量(1 回だけ)
modal run flickr_table1/modal_app.py --stage all --seed 0 --epochs 200      # 5 variant を並列実行
modal run flickr_table1/modal_app.py --stage diagnose                       # random の診断
modal run flickr_table1/modal_app.py --stage coco_features                  # COCO の特徴量(1 回だけ)
modal run flickr_table1/modal_app.py --stage scaling --seed 0               # COCO: good/bad/ntp × 0.6M, 1.7M, 5.8M
modal run flickr_table1/modal_app.py --stage lm                             # Qwen2.5 のベースライン
modal run flickr_table1/modal_app.py --stage train --dataset coco --variant good --samples 100000 --seed 1
```

**動作確認の範囲**: 論文の結果は、すべて Modal の A10G (24GB) で出したものです。ローカル版は、VRAM 12GB のローカル GPU で COCO 512 枚の試運転(特徴量の抽出と、2,560 サンプルの学習)が最後まで動くことだけを確認しました。


### 結果(seed 0、200 epoch、テスト 1,000 組)

| variant                     | mutual 10-NN | R@1      | R@5      |
| --------------------------- | ------------ | -------- | -------- |
| good view                   | **.196**     | **.073** | **.212** |
| bad view                    | .087         | .043     | .107     |
| Next Token Prediction       | .094         | .022     | .066     |
| Next Token Prediction(ペア) | .097         | .022     | .063     |
| random(未学習)              | .033         | .036     | .075     |

結果は `flickr_table1/results/*_seed0_ep200.json` に保存してあります。

### 論文との比較

- 傾向は同じです。good view が 3 指標すべてで最高になり、Next Token Prediction をペアで学習しても改善しません。
- 絶対値は論文と比べられません。論文の Table 1 は good view が mutual 10-NN .062、R@1 .044 ですが、画像側が違います。一致したと言えるのは傾向だけです。
- seed が 1 つなので、小さい差は断定できません。bad view と Next Token Prediction の mutual 10-NN(.087 と .094)は順位が逆です。

### 気づいた点

- 未学習の random でも、R@1 が .036 あり、偶然の水準(.001)を大きく上回ります。対応をシャッフルすると .001〜.002 に戻るので、リークではありません。Transformer を通さない単語埋め込みの平均だけでも同程度(.033〜.044)なので、原因は「同じ単語を含む文が近くに来る」ことと、それが画像の内容と相関することだと考えています。論文の random init(R@1 .001)と水準が違う理由は特定できていません。
- Next Token Prediction の R@1 と R@5 は、未学習の random を下回ります。mutual 10-NN は上がっているので、近傍構造は学習していますが、1 つの直交写像で画像と対応づける指標では、単語の重なりという下限を超えていません。
- good view の検証 mutual 10-NN は、epoch 50 の .212 から最終の .184 まで下がりました。29,000 画像で長く学習した影響かもしれませんが、確認していません。

## COCO でのスケーリング実験(論文の主実験の縮小版)

論文の主実験は、サンプル数と整合度の関係(スケーリング則)を調べるものです。同じ構成を COCO(Karpathy 分割。train 113,287、val 5,000、test 5,000)に縮小しました。text 側のモデル、画像側(凍結した DINOv2 ViT-B/14)、目的関数、評価は Flickr30k の実験と同じです。変えたのは学習で見るサンプル数で、1 サンプルは 1 画像のグループ(good view なら 8 ビュー)です。コードは `flickr_table1/run.py` の `--dataset coco --samples N` です(Modal なら `modal_app.py` の `--stage train`)。

### 結果

テスト 5,000 組の mutual 10-NN です(偶然の水準は .002)。good view と Next Token Prediction は 5 万〜170 万サンプルを seed 3 本(0, 1, 2)で、平均 ± 標準偏差を示します。bad view と 580 万サンプルの点は seed 0 の 1 本だけです。

| サンプル数 | good view   | Next Token Prediction | bad view |
| ---------- | ----------- | --------------------- | -------- |
| 5 万       | .139 ± .002 | .025 ± .007           |          |
| 10 万      | .148 ± .001 | .068 ± .000           |          |
| 30 万      | .164 ± .001 | .092 ± .001           |          |
| 60 万      | .173 ± .001 | .099 ± .001           | .079     |
| 170 万     | .182 ± .001 | .103 ± .002           | .076     |
| 580 万     | .181        | .097                  | .071     |

未学習(random)は .049 です。図は `flickr_table1/results/scaling_coco.svg` にあります(`python flickr_table1/make_figure.py` で再生成できます)。

R@1 は、good view が .057(5 万)から .075(170 万)、Next Token Prediction が .040 から .048、未学習が .042 でした。R@1 は点ごとの揺れが大きく、good view の 30 万(.062)は 10 万(.065)より低くなっています。そのため、傾向の判断は mutual 10-NN を主にしています。

凍結した Qwen2.5 を同じ DINOv2 特徴量と比べた結果も示します。層は検証セットの mutual 10-NN で選びました。

| モデル       | 最良層 | mutual 10-NN | R@1  |
| ------------ | ------ | ------------ | ---- |
| Qwen2.5-0.5B | 1      | .095         | .053 |
| Qwen2.5-1.5B | 27     | .084         | .050 |
| Qwen2.5-3B   | 33     | .087         | .049 |
| Qwen2.5-7B   | 27     | .087         | .054 |

### 論文との比較

- good view は、5 万〜170 万サンプルの範囲で、対数軸に対してほぼ直線に伸びました。10 倍あたり約 +.028 で、当てはめの R² は .986 です。seed の標準偏差(.001〜.002)に比べて、点の差は十分に大きいです。
- 170 万サンプルを超えると、伸びが止まります(580 万で .181)。論文のスケーリング則は、もっと広い範囲で続くと主張しているので、この点は一致していません。飽和の原因は、同じ約 11 万画像を 50 周以上見ていることだと考えていますが、確かめていません。
- Next Token Prediction は、5 万サンプルで未学習(.049)を下回り、30 万〜60 万で .09〜.10 に達してから飽和します。5 万〜60 万の傾きは 10 倍あたり約 +.066 ですが、直線への当てはまりは good view より悪いです(R² .887)。
- 小さい JEPA が大きい LM を上回る点は、再現できました。38M パラメータの good view(.139 以上、60 万サンプルで .173)が、凍結した Qwen2.5-7B(.087)を上回ります。ただし、LM 側は層ごとの平均プーリング特徴で、論文の評価条件と同じかは確認していません。
- 「Next Token Prediction の約 1/42 のサンプルで同じ整合に達する」は、倍率を決められません。Next Token Prediction の最高値(.103)に、good view は 5 万〜10 万サンプルの間で届きます。ただし、good view の最低点(5 万で .139)が Next Token Prediction の最高値より高いため、同じ整合度で比べられる点がありません。
- bad view は、サンプル数を増やしても改善しません(60 万の .079 から 580 万の .071 へ、むしろ下がっています)。Thm. 3 の「ニュイサンスの分だけ整合が下がる」とは矛盾しません。

### 限界

- 画像側が凍結した DINOv2 なので、整合度の上限は DINOv2 の表現で決まります。論文の絶対値とは比べられません。
- bad view と 580 万サンプルの点は seed 1 本です。
- 論文の 85 run で測っている「good view の割合」の軸は調べていません。
- LM は Qwen2.5-7B までで、論文の 72B とは規模が違います。
- 5,000 組の評価では、R@1 が小さい値(.03〜.08)になるため、細かい差は見分けられません。
- 学習率などは Flickr30k と同じ値を使っており、サンプル数ごとの調整はしていません。

## 所感・疑問点

- Thm. 1 の証明は、各 f_m∘g_m が同じ L_π の最小解になるというものです。そのため主張の強さは、最小解が回転を除いて一意であることに依存します。Prop. 1 の gap 条件が実データでどの程度成り立つかが気になります。
- 論文自身も、SIGReg のような分布全体の制約のもとでの一意性は仮定として残しています。
- 論文は、画像側の良い good view(同一シーンの複数観測、たとえば動画のフレーム)のスケールアップを今後の課題としています。

## 参考

- 論文 PDF: arXiv:2610.39302
