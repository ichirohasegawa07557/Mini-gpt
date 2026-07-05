#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "minigpt/model.h"
#include "minigpt/train.h"
#include "minigpt/generate.h"
#include "minigpt/viz.h"

#define CORPUS "data/tiny_corpus.txt"
#define OUTDIR "results"
#define CKPT   "results/mini_gpt_checkpoint.bin"

static GPTConfig default_gcfg(void) {
    GPTConfig g; g.vocab_size = 0; g.block_size = 64;
    g.n_layer = 2; g.n_head = 2; g.n_embd = 64; return g;
}
static TrainConfig default_tcfg(void) {
    TrainConfig t; t.max_iters = 200; t.batch_size = 16; t.learning_rate = 3e-3;
    t.eval_interval = 25; t.grad_clip = 1.0; t.weight_decay = 0.01; t.seed = 42;
    return t;
}

static int cmd_train(int iters) {
    MiniGPT m; CharTokenizer tok;
    HistRow hist[1024]; int hn = 0;
    GPTConfig g = default_gcfg(); TrainConfig t = default_tcfg();
    if (iters > 0) t.max_iters = iters;
    if (train_char_model(CORPUS, OUTDIR, g, t, &m, &tok, hist, &hn) != 0) return 1;
    viz_training_curve_svg(OUTDIR "/training_curve.svg", hist, hn);
    printf("saved: %s, %s/training_history.csv, %s/training_curve.svg\n",
           CKPT, OUTDIR, OUTDIR);
    model_free(&m);
    return 0;
}

static int cmd_generate(const char *prompt, int n_new) {
    MiniGPT m; CharTokenizer tok; Rng rng;
    char *out;
    FILE *f;
    if (model_load(&m, &tok, CKPT) != 0) {
        fprintf(stderr, "no checkpoint found — run './bin/minigpt train' first\n");
        return 1;
    }
    rng_seed(&rng, 123);
    out = (char *)malloc(strlen(prompt) + n_new + 2);
    if (generate_text(&m, &tok, &rng, prompt, n_new, 0.8, 10, out,
                      (long)strlen(prompt) + n_new + 2) < 0) { free(out); model_free(&m); return 1; }
    printf("--- generated text ---\n%s\n", out);
    f = fopen(OUTDIR "/generated_text.txt", "w");
    if (f) { fprintf(f, "%s\n", out); fclose(f); }
    free(out); model_free(&m);
    return 0;
}

static int cmd_attention(const char *prompt) {
    MiniGPT m; CharTokenizer tok;
    int ids[512], T;
    double *logits;
    if (model_load(&m, &tok, CKPT) != 0) {
        fprintf(stderr, "no checkpoint found — run './bin/minigpt train' first\n");
        return 1;
    }
    T = (int)strlen(prompt);
    if (T > m.cfg.block_size) T = m.cfg.block_size;
    if (tokenizer_encode(&tok, prompt, T, ids) != 0) {
        fprintf(stderr, "prompt contains characters outside the corpus vocabulary\n");
        model_free(&m);
        return 1;
    }
    logits = (double *)malloc(sizeof(double) * (long)T * m.cfg.vocab_size);
    model_forward(&m, ids, T, logits);
    {
        /* copy the [T x T] window out of the [block x block] cache */
        double *att = (double *)malloc(sizeof(double) * (long)T * T);
        int a, b;
        for (a = 0; a < T; ++a)
            for (b = 0; b < T; ++b)
                att[(long)a * T + b] = m.last_att[(long)a * m.cfg.block_size + b];
        viz_attention_map_svg(OUTDIR "/attention_map.svg", att, T, prompt);
        free(att);
    }
    printf("saved: %s/attention_map.svg (prompt: \"%s\")\n", OUTDIR, prompt);
    free(logits); model_free(&m);
    return 0;
}

static int cmd_structure(void) {
    MiniGPT m; CharTokenizer tok; Rng rng;
    FILE *f;
    if (model_load(&m, &tok, CKPT) != 0) {
        GPTConfig g = default_gcfg();
        g.vocab_size = 36;  /* vocab of the bundled corpus; retrain to update */
        rng_seed(&rng, 42);
        model_init(&m, g, &rng);
        printf("(no checkpoint: showing structure for the default config)\n\n");
    }
    model_print_structure(&m, stdout);
    f = fopen(OUTDIR "/model_structure.txt", "w");
    if (f) { model_print_structure(&m, f); fclose(f); }
    printf("\nsaved: %s/model_structure.txt\n", OUTDIR);
    model_free(&m);
    return 0;
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "all";
    if (strcmp(cmd, "train") == 0)
        return cmd_train(argc > 2 ? atoi(argv[2]) : 0);
    if (strcmp(cmd, "generate") == 0)
        return cmd_generate(argc > 2 ? argv[2] : "The ",
                            argc > 3 ? atoi(argv[3]) : 200);
    if (strcmp(cmd, "attention") == 0)
        return cmd_attention(argc > 2 ? argv[2] : "To be, or not to be");
    if (strcmp(cmd, "structure") == 0)
        return cmd_structure();
    if (strcmp(cmd, "all") == 0) {
        if (cmd_train(0)) return 1;
        if (cmd_structure()) return 1;
        if (cmd_generate("The ", 200)) return 1;
        return cmd_attention("To be, or not to be");
    }
    fprintf(stderr,
        "usage: %s [command]\n"
        "  structure            print the model architecture tree\n"
        "  train [iters]        train on data/tiny_corpus.txt (default 200 iters)\n"
        "  generate [p] [n]     sample n chars from prompt p (temp 0.8, top-k 10)\n"
        "  attention [prompt]   save attention heatmap SVG for the prompt\n"
        "  all                  train + structure + generate + attention\n", argv[0]);
    return 2;
}
