#include "minigpt/train.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void adamw_init(AdamW *o, long n, double lr, double weight_decay) {
    o->m = (double *)calloc(n, sizeof(double));
    o->v = (double *)calloc(n, sizeof(double));
    o->n = n; o->lr = lr; o->beta1 = 0.9; o->beta2 = 0.999;
    o->eps = 1e-8; o->weight_decay = weight_decay; o->step = 0;
}
void adamw_free(AdamW *o) { free(o->m); free(o->v); }

void adamw_step(AdamW *o, double *params, double *grads) {
    long i;
    double bc1, bc2;
    o->step++;
    bc1 = 1.0 - pow(o->beta1, (double)o->step);
    bc2 = 1.0 - pow(o->beta2, (double)o->step);
    for (i = 0; i < o->n; ++i) {
        double g = grads[i];
        o->m[i] = o->beta1 * o->m[i] + (1.0 - o->beta1) * g;
        o->v[i] = o->beta2 * o->v[i] + (1.0 - o->beta2) * g * g;
        /* decoupled weight decay, as in torch.optim.AdamW */
        params[i] -= o->lr * o->weight_decay * params[i];
        params[i] -= o->lr * (o->m[i] / bc1) / (sqrt(o->v[i] / bc2) + o->eps);
    }
}

double grad_clip_global_norm(double *grads, long n, double max_norm) {
    long i; double s = 0;
    for (i = 0; i < n; ++i) s += grads[i] * grads[i];
    s = sqrt(s);
    if (s > max_norm && s > 0) {
        double k = max_norm / s;
        for (i = 0; i < n; ++i) grads[i] *= k;
    }
    return s;
}

static char *read_file(const char *path, long *out_n) {
    FILE *f = fopen(path, "rb");
    long n; char *buf;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    buf = (char *)malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    buf[n] = '\0'; fclose(f);
    *out_n = n;
    return buf;
}

/* random contiguous (x, y=x shifted by 1) pair — mirrors get_batch */
static void sample_xy(const int *data, long n, int T, Rng *rng, int *x, int *y) {
    long i0 = rng_randint(rng, (int)(n - T - 1));
    int t;
    for (t = 0; t < T; ++t) { x[t] = data[i0 + t]; y[t] = data[i0 + t + 1]; }
}

static double estimate_loss(MiniGPT *m, const int *data, long n, int B, int T, Rng *rng) {
    int *x = (int *)malloc(sizeof(int) * T), *y = (int *)malloc(sizeof(int) * T);
    double s = 0; int b;
    for (b = 0; b < B; ++b) { sample_xy(data, n, T, rng, x, y); s += model_loss_only(m, x, y, T); }
    free(x); free(y);
    return s / B;
}

int train_char_model(const char *corpus_path, const char *out_dir,
                     GPTConfig gcfg, TrainConfig tcfg,
                     MiniGPT *out_model, CharTokenizer *out_tok,
                     HistRow *hist, int *hist_len) {
    long n_text = 0, n_train, n_val;
    char *text = read_file(corpus_path, &n_text);
    char path[512];
    int *data, *x, *y, step, b, hn = 0;
    const int *train_d, *val_d;
    Rng rng;
    AdamW opt;
    FILE *f;
    int val_is_train = 0;

    if (!text) { fprintf(stderr, "cannot read corpus: %s\n", corpus_path); return -1; }
    rng_seed(&rng, tcfg.seed);
    tokenizer_from_text(out_tok, text, n_text);
    gcfg.vocab_size = out_tok->vocab_size;

    data = (int *)malloc(sizeof(int) * n_text);
    tokenizer_encode(out_tok, text, n_text, data);
    n_train = (long)(n_text * 0.9);
    n_val = n_text - n_train;
    train_d = data; val_d = data + n_train;
    if (n_train <= gcfg.block_size + 1) {
        fprintf(stderr, "corpus too short for block_size=%d\n", gcfg.block_size);
        free(text); free(data); return -1;
    }
    if (n_val <= gcfg.block_size + 1) {   /* FIX: report the fallback explicitly */
        val_is_train = 1; val_d = train_d; n_val = n_train;
        printf("note: validation split (%ld chars) is smaller than block_size+1;\n"
               "      val_loss is computed on the training split (as in the original,\n"
               "      but now reported instead of silent).\n", n_text - n_train);
    }

    model_init(out_model, gcfg, &rng);
    adamw_init(&opt, out_model->n_params, tcfg.learning_rate, tcfg.weight_decay);
    x = (int *)malloc(sizeof(int) * gcfg.block_size);
    y = (int *)malloc(sizeof(int) * gcfg.block_size);

    printf("training: %d iters, batch %d, block %d, vocab %d, params %ld\n",
           tcfg.max_iters, tcfg.batch_size, gcfg.block_size,
           gcfg.vocab_size, out_model->n_params);

    /* FIX of the original off-by-one: the PyTorch loop ran max_iters+1 update
     * steps and logged the "final" loss before the last update. Here we run
     * exactly max_iters updates and log AFTER updating, so the last row of
     * the history reflects the final weights. */
    for (step = 1; step <= tcfg.max_iters; ++step) {
        double tl = 0;
        model_zero_grads(out_model);
        for (b = 0; b < tcfg.batch_size; ++b) {
            sample_xy(train_d, n_train, gcfg.block_size, &rng, x, y);
            tl += model_loss_and_backward(out_model, x, y, gcfg.block_size,
                                          tcfg.batch_size);
        }
        grad_clip_global_norm(out_model->grads, out_model->n_params, tcfg.grad_clip);
        adamw_step(&opt, out_model->params, out_model->grads);

        if (step % tcfg.eval_interval == 0 || step == 1 || step == tcfg.max_iters) {
            double vl = estimate_loss(out_model, val_d, n_val, 4, gcfg.block_size, &rng);
            hist[hn].step = step;
            hist[hn].train_loss = tl / tcfg.batch_size;
            hist[hn].val_loss = vl;
            printf("step %4d  train_loss %.4f  val_loss %.4f%s\n",
                   step, hist[hn].train_loss, vl, val_is_train ? " (val=train)" : "");
            hn++;
        }
    }
    *hist_len = hn;

    snprintf(path, sizeof(path), "%s/training_history.csv", out_dir);
    f = fopen(path, "w");
    if (f) {
        fprintf(f, "step,train_loss,val_loss\n");
        for (b = 0; b < hn; ++b)
            fprintf(f, "%d,%.6f,%.6f\n", hist[b].step, hist[b].train_loss, hist[b].val_loss);
        fclose(f);
    }
    snprintf(path, sizeof(path), "%s/model_metadata.txt", out_dir);
    f = fopen(path, "w");
    if (f) {
        fprintf(f, "vocab_size=%d\nparameter_count=%ld\nmax_iters=%d\nblock_size=%d\n"
                   "val_split_usable=%s\nimplementation=pure_C99_no_external_libraries\n",
                gcfg.vocab_size, out_model->n_params, tcfg.max_iters,
                gcfg.block_size, val_is_train ? "no_fell_back_to_train" : "yes");
        fclose(f);
    }
    snprintf(path, sizeof(path), "%s/mini_gpt_checkpoint.bin", out_dir);
    model_save(out_model, out_tok, path);

    adamw_free(&opt);
    free(text); free(data); free(x); free(y);
    return 0;
}
