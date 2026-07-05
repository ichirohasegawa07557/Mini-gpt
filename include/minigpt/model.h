#ifndef MINIGPT_MODEL_H
#define MINIGPT_MODEL_H

#include <stdio.h>
#include "minigpt/config.h"
#include "minigpt/rng.h"
#include "minigpt/tokenizer.h"

/*
 * MiniGPT — mirrors src/model.py:
 *   token_emb + pos_emb
 *   -> n_layer x Block( x + Attn(LN1(x)) ; x + MLP(LN2(x)) )
 *   -> LN_f -> head -> logits
 * All parameters live in one flat double array (model->params) described
 * by the offset table below, so AdamW/clipping/serialization stay trivial.
 */
typedef struct {
    long ln1_g, ln1_b;      /* [C], [C]        */
    long qkv_w, qkv_b;      /* [3C x C], [3C]  */
    long proj_w, proj_b;    /* [C x C],  [C]   */
    long ln2_g, ln2_b;      /* [C], [C]        */
    long fc1_w, fc1_b;      /* [4C x C], [4C]  */
    long fc2_w, fc2_b;      /* [C x 4C], [C]   */
} BlockOffsets;

typedef struct {
    GPTConfig cfg;
    long n_params;
    double *params;         /* flat parameter vector */
    double *grads;          /* same layout as params */
    long tok_emb;           /* [V x C]   offset into the flat vector */
    long pos_emb;           /* [Tmax x C] */
    BlockOffsets *blk;      /* per-layer offsets */
    long lnf_g, lnf_b;      /* [C], [C] */
    long head_w, head_b;    /* [V x C], [V] */
    /* activation cache for backward (one sequence, length <= block_size) */
    double *acts;
    long   n_acts;
    /* last attention of block 0 / head 0, [T x T], for visualization */
    double *last_att;
    int    last_att_T;
} MiniGPT;

void   model_init(MiniGPT *m, GPTConfig cfg, Rng *rng);
void   model_free(MiniGPT *m);
/* forward one sequence idx[T]; fills logits[T x V]; caches activations */
void   model_forward(MiniGPT *m, const int *idx, int T, double *logits);
/* mean cross-entropy for one sequence; grads are accumulated already
 * normalized by T and by n_batch (pass the batch size; 1 for a single
 * sequence), matching F.cross_entropy's mean over B*T. */
double model_loss_and_backward(MiniGPT *m, const int *idx, const int *targets,
                               int T, int n_batch);
double model_loss_only(MiniGPT *m, const int *idx, const int *targets, int T);
void   model_zero_grads(MiniGPT *m);
void   model_print_structure(const MiniGPT *m, FILE *fout);
int    model_save(const MiniGPT *m, const CharTokenizer *tok, const char *path);
int    model_load(MiniGPT *m, CharTokenizer *tok, const char *path);

#endif
