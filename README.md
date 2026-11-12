# Papers

読んだ論文の要約と、その実装コードをまとめるリポジトリです。

## 論文一覧

分野ごとに分けています。各分野の `▶` をクリックすると、表を開閉できます。

- [RAG と検索](#rag-と検索)(3 本): RAG、reranking、ファインチューニング、法務への応用
- [表現学習](#表現学習)(1 本): JEPA、マルチモーダルの表現

### RAG と検索

<details open>
<summary>3 本(クリックで開閉)</summary>

| 年 | タイトル | タグ | メモ・実装 |
|----|----------|------|-----------|
| 2026 | [Using LLMs for Legal Decision-Making in Austrian VAT Law](2026-vat-llm/) | RAG, fine-tuning, 法務, VAT | 要約 + 小規模検証 (numpy) + 縮小再現 (Modal/Qwen2.5-0.5B、合成ドイツ語事例) + 日本の消費税(内外判定)での検証 (Qwen2.5-1.5B/3B、検索の比較) |
| 2024 | [RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs](2024-rankrag/) | RAG, reranking, instruction-tuning | 要約 + 小規模実装 (numpy) + 縮小再現 2 種 (Modal/Qwen2.5。SQuAD・MS MARCO と、gold 文脈つき HotpotQA) |
| 2024 | [RAFT: Adapting Language Model to Domain Specific RAG](2024-raft/) | RAG, ドメイン適応, ファインチューニング | 要約 + 小規模検証 (numpy) + 縮小再現 (Modal/Qwen2.5-0.5B) |

</details>

### 表現学習

<details open>
<summary>1 本(クリックで開閉)</summary>

| 年 | タイトル | タグ | メモ・実装 |
|----|----------|------|-----------|
| 2026 | [Proving and Improving the Platonic Representation Hypothesis with JEPAs](2026-platonic-representation-jepa/) | JEPA, 表現学習, マルチモーダル | 要約 + 理論の数値検証 (numpy) + Flickr30k / COCO での縮小実験 (PyTorch, Modal) |

</details>

## 構成

各論文は `<年>-<短縮タイトル>/` に置きます。新しい論文は `_template/` をコピーして始め、上の一覧の該当する分野の表に 1 行足します (該当する分野がなければ、`###` の見出しと `<details>` のブロックを足します)。

```
<年>-<短縮タイトル>/
├── README.md   # 要約・手法・実験結果・所感
└── src/        # 実装コード
```
