#ifndef MINIGPT_CONFIG_H
#define MINIGPT_CONFIG_H

/* Model hyperparameters — mirrors GPTConfig in the original PyTorch code. */
typedef struct {
    int vocab_size;   /* determined from the corpus at runtime */
    int block_size;   /* context length (default 64) */
    int n_layer;      /* transformer blocks (default 2) */
    int n_head;       /* attention heads (default 2) */
    int n_embd;       /* embedding width (default 64) */
} GPTConfig;

typedef struct {
    int    max_iters;      /* default 200 */
    int    batch_size;     /* default 16 */
    double learning_rate;  /* default 3e-3 */
    int    eval_interval;  /* default 25 */
    double grad_clip;      /* default 1.0 (global norm) */
    double weight_decay;   /* default 0.01, decoupled (AdamW) */
    unsigned long long seed;
} TrainConfig;

#endif
