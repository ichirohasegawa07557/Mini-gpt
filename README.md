# Mini GPT in Pure C

順伝播だけでなく、逆伝播(自動微分なしの手書きバックプロパゲーション)・AdamW・生成・可視化をCで実装しています。

## ディレクトリ構造

```
mini-gpt-c/
├── Makefile
├── README.md
├── data/
│   └── tiny_corpus.txt          # 元リポジトリと同一のコーパス (504文字, 語彙36)
├── include/minigpt/             # 1コンポーネント = 1ヘッダ(構造がそのまま見える)
│   ├── config.h                 #   GPTConfig / TrainConfig
│   ├── rng.h                    #   xorshift64* + Box-Muller (rand()不使用)
│   ├── tokenizer.h              #   文字レベルトークナイザ
│   ├── model.h                  #   MiniGPT本体(順伝播+逆伝播)
│   ├── train.h                  #   AdamW / 勾配クリップ / 学習ループ
│   ├── generate.h               #   temperature + top-k サンプリング
│   └── viz.h                    #   SVG出力(matplotlib代替)
├── src/
│   ├── rng.c  tokenizer.c  model.c  train.c  generate.c  viz.c
│   └── main.c                   # CLI: structure / train / generate / attention / all
├── tests/
│   └── test_minigpt.c           # 勾配の数値検証を含む8テスト
└── results/                     # 生成物(git管理外)
```

## モデル構造

```
tokens ──> token_emb + pos_emb ──> [Block × n_layer] ──> LN_f ──> head ──> logits
                                      │
              Block:  x ── LN1 ── CausalSelfAttention ──(+)── LN2 ── MLP(4x, GELU) ──(+)──> x'
                      └──────────────residual──────────┘ └──────────residual────────┘
```

`./bin/minigpt structure` で各層の形状とパラメータ数のツリーを表示します(既定構成: 2層・2ヘッド・幅64・語彙36で **108,836 パラメータ**)。

## ビルドと実行

```bash
make all          # bin/minigpt と bin/tests をビルド(依存: ccとlibmのみ)
make test         # 8テスト実行(勾配検証を含む)
./bin/minigpt all # 学習(200 iter) → 構造表示 → 生成 → アテンション可視化
```

個別コマンド:

```bash
./bin/minigpt structure
./bin/minigpt train 200
./bin/minigpt generate "The " 200
./bin/minigpt attention "To be, or not to be"
```

生成物: `results/mini_gpt_checkpoint.bin`, `training_history.csv`, `training_curve.svg`, `generated_text.txt`, `attention_map.svg`, `model_structure.txt`

## 正しさの担保

手書き逆伝播は誤りやすいため、テストで**中心差分による数値微分と解析勾配を60個のランダムパラメータで比較**しています(最大相対誤差 ~3e-06)。全計算はdouble精度です。ほかに、トークナイザの往復、初期損失 ≈ ln(vocab)、アテンションの因果性(未来を見ない・行和=1)、AdamWによる損失低下、生成長の検証を行います。

## 元実装からの訂正点

1. **学習ループのoff-by-one**: 元の `for step in range(max_iters+1)` は max_iters+1 回更新し、履歴の最終行が最後の更新**前**の損失を記録していました。本実装は正確に max_iters 回更新し、更新**後**に記録します。
2. **検証データの無言フォールバック**: 付属コーパス(504文字)では検証分割(約50文字)が block_size+1=65 未満のため、元実装は val_loss を訓練データで計算しながらその旨を表示しませんでした。本実装は同じフォールバックを行いつつ `(val=train)` と明示します。
3. **READMEの構成図の不一致**: 元READMEのツリーはリポジトリに存在する `notebooks/` を含まず、ルート名も実際のリポジトリ名と異なっていました。本READMEは実体と一致させています。

## PyTorch版との意図的な差異

- **dropout**: 元実装の0.1に対し本実装は使用しません(小コーパスの教育実装では決定的な挙動を優先。勾配検証も厳密になります)。
- **GELU**: PyTorchの既定(erf)ではなくtanh近似(GPT-2と同じ)。
- **checkpoint**: `.pt` の代わりに自前バイナリ形式 `.bin`(magic + config + 語彙 + パラメータ)。
- **可視化**: matplotlib/Streamlitの代わりに自前SVG出力。

## 制限

元実装と同じく教育目的です。文字レベル・極小コーパスのため、生成文はコーパス風の綴りを再現する程度の品質です。
