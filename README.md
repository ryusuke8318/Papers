# Papers

読んだ論文の要約と、その実装コードをまとめるリポジトリです。

## 論文一覧

| 年 | タイトル | タグ | メモ・実装 |
|----|----------|------|-----------|
| 2026 | [Proving and Improving the Platonic Representation Hypothesis with JEPAs](2026-platonic-representation-jepa/) | JEPA, 表現学習, マルチモーダル | 要約 + 理論の数値検証 (numpy) + Flickr30k / COCO での縮小実験 (PyTorch, Modal) |
| 2024 | [RAFT: Adapting Language Model to Domain Specific RAG](2024-raft/) | RAG, ドメイン適応, ファインチューニング | 要約 + 小規模検証 (numpy) + 縮小再現 (Modal/Qwen2.5-0.5B) |

## 構成

各論文は `<年>-<短縮タイトル>/` に置きます。新しい論文は `_template/` をコピーして始めます。

```
<年>-<短縮タイトル>/
├── README.md   # 要約・手法・実験結果・所感
└── src/        # 実装コード
```
