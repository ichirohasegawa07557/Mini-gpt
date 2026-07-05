#ifndef MINIGPT_TRAIN_H
#define MINIGPT_TRAIN_H

#include "minigpt/model.h"

/* AdamW state (decoupled weight decay), mirrors torch.optim.AdamW */
typedef struct {
    double *m, *v;
    long   n;
    double lr, beta1, beta2, eps, weight_decay;
    long   step;
} AdamW;

void   adamw_init(AdamW *o, long n, double lr, double weight_decay);
void   adamw_free(AdamW *o);
void   adamw_step(AdamW *o, double *params, double *grads);
double grad_clip_global_norm(double *grads, long n, double max_norm);

typedef struct { int step; double train_loss, val_loss; } HistRow;

/* Full training loop — mirrors train_char_model (src/train.py).
 * Writes results/training_history.csv, results/model_metadata.txt,
 * results/mini_gpt_checkpoint.bin and returns the trained model. */
int train_char_model(const char *corpus_path, const char *out_dir,
                     GPTConfig gcfg, TrainConfig tcfg,
                     MiniGPT *out_model, CharTokenizer *out_tok,
                     HistRow *hist, int *hist_len);

#endif
