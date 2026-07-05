#include "minigpt/model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------- small math helpers ---------------- */

static double gelu(double x) {            /* tanh approximation of GELU */
    double c = 0.7978845608028654;        /* sqrt(2/pi) */
    double u = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(u));
}
static double gelu_grad(double x) {
    double c = 0.7978845608028654;
    double u = c * (x + 0.044715 * x * x * x);
    double t = tanh(u);
    double du = c * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du;
}

/* out[T x nout] = in[T x nin] @ W[nout x nin]^T + b[nout] */
static void linear_fwd(double *out, const double *in, const double *W,
                       const double *b, int T, int nin, int nout) {
    int t, o, i;
    for (t = 0; t < T; ++t)
        for (o = 0; o < nout; ++o) {
            double s = b[o];
            const double *w = W + (long)o * nin, *x = in + (long)t * nin;
            for (i = 0; i < nin; ++i) s += w[i] * x[i];
            out[(long)t * nout + o] = s;
        }
}

static void linear_bwd(double *din, double *dW, double *db,
                       const double *dout, const double *in, const double *W,
                       int T, int nin, int nout) {
    int t, o, i;
    if (din) memset(din, 0, sizeof(double) * (long)T * nin);
    for (t = 0; t < T; ++t)
        for (o = 0; o < nout; ++o) {
            double g = dout[(long)t * nout + o];
            const double *x = in + (long)t * nin;
            double *dw = dW + (long)o * nin;
            db[o] += g;
            for (i = 0; i < nin; ++i) {
                dw[i] += g * x[i];
                if (din) din[(long)t * nin + i] += g * W[(long)o * nin + i];
            }
        }
}

static void ln_fwd(double *out, double *mean, double *rstd, const double *in,
                   const double *g, const double *b, int T, int C) {
    int t, i;
    for (t = 0; t < T; ++t) {
        const double *x = in + (long)t * C;
        double mu = 0, var = 0;
        for (i = 0; i < C; ++i) mu += x[i];
        mu /= C;
        for (i = 0; i < C; ++i) { double d = x[i] - mu; var += d * d; }
        var /= C;
        double rs = 1.0 / sqrt(var + 1e-5);
        mean[t] = mu; rstd[t] = rs;
        for (i = 0; i < C; ++i)
            out[(long)t * C + i] = (x[i] - mu) * rs * g[i] + b[i];
    }
}

static void ln_bwd(double *din, double *dg, double *db, const double *dout,
                   const double *in, const double *mean, const double *rstd,
                   const double *g, int T, int C) {
    int t, i;
    for (t = 0; t < T; ++t) {
        const double *x = in + (long)t * C, *dy = dout + (long)t * C;
        double mu = mean[t], rs = rstd[t];
        double s1 = 0, s2 = 0;
        for (i = 0; i < C; ++i) {
            double xh = (x[i] - mu) * rs;
            double dyg = dy[i] * g[i];
            s1 += dyg; s2 += dyg * xh;
            dg[i] += dy[i] * xh;
            db[i] += dy[i];
        }
        s1 /= C; s2 /= C;
        for (i = 0; i < C; ++i) {
            double xh = (x[i] - mu) * rs;
            din[(long)t * C + i] = rs * (dy[i] * g[i] - s1 - xh * s2);
        }
    }
}

/* ---------------- parameter layout ---------------- */

static long alloc_offsets(MiniGPT *m) {
    GPTConfig c = m->cfg;
    long C = c.n_embd, V = c.vocab_size, Tm = c.block_size, off = 0;
    int l;
    m->tok_emb = off; off += V * C;
    m->pos_emb = off; off += Tm * C;
    m->blk = (BlockOffsets *)malloc(sizeof(BlockOffsets) * c.n_layer);
    for (l = 0; l < c.n_layer; ++l) {
        BlockOffsets *b = &m->blk[l];
        b->ln1_g = off; off += C;   b->ln1_b = off; off += C;
        b->qkv_w = off; off += 3 * C * C;  b->qkv_b = off; off += 3 * C;
        b->proj_w = off; off += C * C;     b->proj_b = off; off += C;
        b->ln2_g = off; off += C;   b->ln2_b = off; off += C;
        b->fc1_w = off; off += 4 * C * C;  b->fc1_b = off; off += 4 * C;
        b->fc2_w = off; off += 4 * C * C;  b->fc2_b = off; off += C;
    }
    m->lnf_g = off; off += C;  m->lnf_b = off; off += C;
    m->head_w = off; off += V * C;  m->head_b = off; off += V;
    return off;
}

/* activation cache layout (one sequence of length T = block_size max) */
typedef struct {
    long x0;                       /* [T x C] embeddings in */
    long ln1_out, ln1_mu, ln1_rs;  /* per block, stacked */
    long qkv, att, atty;           /* qkv [T x 3C], att [H x T x T], atty [T x C] */
    long xattn;                    /* [T x C] after attn residual */
    long ln2_out, ln2_mu, ln2_rs;
    long fc1_pre, fc1_act;         /* [T x 4C] each */
    long xout;                     /* [T x C] block output */
} ActOff;

static long act_layout(const MiniGPT *m, ActOff *a) {
    GPTConfig c = m->cfg;
    long C = c.n_embd, T = c.block_size, H = c.n_head, L = c.n_layer, off = 0;
    a->x0 = off; off += T * C;
    a->ln1_out = off; off += L * T * C;
    a->ln1_mu  = off; off += L * T;
    a->ln1_rs  = off; off += L * T;
    a->qkv     = off; off += L * T * 3 * C;
    a->att     = off; off += L * H * T * T;
    a->atty    = off; off += L * T * C;
    a->xattn   = off; off += L * T * C;
    a->ln2_out = off; off += L * T * C;
    a->ln2_mu  = off; off += L * T;
    a->ln2_rs  = off; off += L * T;
    a->fc1_pre = off; off += L * T * 4 * C;
    a->fc1_act = off; off += L * T * 4 * C;
    a->xout    = off; off += L * T * C;
    return off;
}

void model_init(MiniGPT *m, GPTConfig cfg, Rng *rng) {
    long i;
    ActOff a;
    m->cfg = cfg;
    m->n_params = 0; m->blk = NULL;
    m->n_params = alloc_offsets(m);
    m->params = (double *)calloc(m->n_params, sizeof(double));
    m->grads  = (double *)calloc(m->n_params, sizeof(double));
    m->n_acts = act_layout(m, &a);
    m->acts   = (double *)calloc(m->n_acts, sizeof(double));
    m->last_att = (double *)calloc((long)cfg.block_size * cfg.block_size, sizeof(double));
    m->last_att_T = 0;
    /* init: weights ~ N(0, 0.02), biases 0, LayerNorm gains 1 —
       mirrors MiniGPT._init_weights */
    for (i = 0; i < m->n_params; ++i) m->params[i] = rng_normal(rng, 0.0, 0.02);
    {
        int l, j; GPTConfig c = cfg; long C = c.n_embd;
        for (l = 0; l < c.n_layer; ++l) {
            BlockOffsets *b = &m->blk[l];
            for (j = 0; j < C; ++j) {
                m->params[b->ln1_g + j] = 1.0; m->params[b->ln1_b + j] = 0.0;
                m->params[b->ln2_g + j] = 1.0; m->params[b->ln2_b + j] = 0.0;
                m->params[b->proj_b + j] = 0.0; m->params[b->fc2_b + j] = 0.0;
            }
            for (j = 0; j < 3 * C; ++j) m->params[b->qkv_b + j] = 0.0;
            for (j = 0; j < 4 * C; ++j) m->params[b->fc1_b + j] = 0.0;
        }
        for (j = 0; j < C; ++j) { m->params[m->lnf_g + j] = 1.0; m->params[m->lnf_b + j] = 0.0; }
        for (j = 0; j < c.vocab_size; ++j) m->params[m->head_b + j] = 0.0;
    }
}

void model_free(MiniGPT *m) {
    free(m->params); free(m->grads); free(m->acts); free(m->blk); free(m->last_att);
    memset(m, 0, sizeof(*m));
}

void model_zero_grads(MiniGPT *m) {
    memset(m->grads, 0, sizeof(double) * m->n_params);
}

/* ---------------- forward ---------------- */

static void attention_fwd(MiniGPT *m, int l, int T, const double *ln1_out,
                          double *qkv, double *att, double *atty) {
    GPTConfig c = m->cfg;
    int C = c.n_embd, H = c.n_head, hd = C / H, h, t, t2, i;
    double scale = 1.0 / sqrt((double)hd);
    linear_fwd(qkv, ln1_out, m->params + m->blk[l].qkv_w,
               m->params + m->blk[l].qkv_b, T, C, 3 * C);
    for (h = 0; h < H; ++h) {
        for (t = 0; t < T; ++t) {
            double *row = att + ((long)h * T + t) * T;
            double mx = -1e30;
            for (t2 = 0; t2 <= t; ++t2) {          /* causal mask */
                const double *q = qkv + (long)t * 3 * C + h * hd;
                const double *k = qkv + (long)t2 * 3 * C + C + h * hd;
                double s = 0;
                for (i = 0; i < hd; ++i) s += q[i] * k[i];
                row[t2] = s * scale;
                if (row[t2] > mx) mx = row[t2];
            }
            double denom = 0;
            for (t2 = 0; t2 <= t; ++t2) { row[t2] = exp(row[t2] - mx); denom += row[t2]; }
            for (t2 = 0; t2 <= t; ++t2) row[t2] /= denom;
            for (t2 = t + 1; t2 < T; ++t2) row[t2] = 0.0;
            for (i = 0; i < hd; ++i) {
                double s = 0;
                for (t2 = 0; t2 <= t; ++t2)
                    s += row[t2] * qkv[(long)t2 * 3 * C + 2 * C + h * hd + i];
                atty[(long)t * C + h * hd + i] = s;
            }
        }
    }
    if (l == 0) {  /* keep block0/head0 for the attention-map viz */
        for (t = 0; t < T; ++t)
            for (t2 = 0; t2 < T; ++t2)
                m->last_att[(long)t * c.block_size + t2] = att[(long)t * T + t2];
        m->last_att_T = T;
    }
}

void model_forward(MiniGPT *m, const int *idx, int T, double *logits) {
    GPTConfig c = m->cfg;
    int C = c.n_embd, V = c.vocab_size, l, t, i;
    ActOff a; act_layout(m, &a);
    double *A = m->acts;
    double *x = A + a.x0;
    for (t = 0; t < T; ++t)
        for (i = 0; i < C; ++i)
            x[(long)t * C + i] = m->params[m->tok_emb + (long)idx[t] * C + i]
                               + m->params[m->pos_emb + (long)t * C + i];
    for (l = 0; l < c.n_layer; ++l) {
        BlockOffsets *b = &m->blk[l];
        long TC = (long)c.block_size * C;
        double *ln1o = A + a.ln1_out + l * TC;
        double *qkv  = A + a.qkv + (long)l * c.block_size * 3 * C;
        double *att  = A + a.att + (long)l * c.n_head * c.block_size * c.block_size;
        double *atty = A + a.atty + l * TC;
        double *xat  = A + a.xattn + l * TC;
        double *ln2o = A + a.ln2_out + l * TC;
        double *f1p  = A + a.fc1_pre + (long)l * c.block_size * 4 * C;
        double *f1a  = A + a.fc1_act + (long)l * c.block_size * 4 * C;
        double *xo   = A + a.xout + l * TC;
        double *xin  = (l == 0) ? x : A + a.xout + (long)(l - 1) * TC;

        ln_fwd(ln1o, A + a.ln1_mu + (long)l * c.block_size,
               A + a.ln1_rs + (long)l * c.block_size, xin,
               m->params + b->ln1_g, m->params + b->ln1_b, T, C);
        attention_fwd(m, l, T, ln1o, qkv, att, atty);
        /* proj + residual */
        {
            double *tmp = (double *)malloc(sizeof(double) * (long)T * C);
            linear_fwd(tmp, atty, m->params + b->proj_w, m->params + b->proj_b, T, C, C);
            for (t = 0; t < T * C; ++t) xat[t] = xin[t] + tmp[t];
            free(tmp);
        }
        ln_fwd(ln2o, A + a.ln2_mu + (long)l * c.block_size,
               A + a.ln2_rs + (long)l * c.block_size, xat,
               m->params + b->ln2_g, m->params + b->ln2_b, T, C);
        linear_fwd(f1p, ln2o, m->params + b->fc1_w, m->params + b->fc1_b, T, C, 4 * C);
        for (t = 0; t < T * 4 * C; ++t) f1a[t] = gelu(f1p[t]);
        {
            double *tmp = (double *)malloc(sizeof(double) * (long)T * C);
            linear_fwd(tmp, f1a, m->params + b->fc2_w, m->params + b->fc2_b, T, 4 * C, C);
            for (t = 0; t < T * C; ++t) xo[t] = xat[t] + tmp[t];
            free(tmp);
        }
    }
    /* final LN + head — reuse ln buffers appended after blocks via locals */
    {
        double *xf = (c.n_layer == 0) ? x
                   : A + a.xout + (long)(c.n_layer - 1) * c.block_size * C;
        double *lnf = (double *)malloc(sizeof(double) * (long)T * C);
        double *mu = (double *)malloc(sizeof(double) * T);
        double *rs = (double *)malloc(sizeof(double) * T);
        ln_fwd(lnf, mu, rs, xf, m->params + m->lnf_g, m->params + m->lnf_b, T, C);
        linear_fwd(logits, lnf, m->params + m->head_w, m->params + m->head_b, T, C, V);
        free(lnf); free(mu); free(rs);
    }
}

/* ---------------- loss + backward ---------------- */

double model_loss_only(MiniGPT *m, const int *idx, const int *targets, int T) {
    int V = m->cfg.vocab_size, t, v;
    double *logits = (double *)malloc(sizeof(double) * (long)T * V);
    double loss = 0;
    model_forward(m, idx, T, logits);
    for (t = 0; t < T; ++t) {
        double *row = logits + (long)t * V, mx = row[0], denom = 0;
        for (v = 1; v < V; ++v) if (row[v] > mx) mx = row[v];
        for (v = 0; v < V; ++v) denom += exp(row[v] - mx);
        loss += -(row[targets[t]] - mx - log(denom));
    }
    free(logits);
    return loss / T;
}

double model_loss_and_backward(MiniGPT *m, const int *idx, const int *targets,
                               int T, int n_batch) {
    GPTConfig c = m->cfg;
    int C = c.n_embd, V = c.vocab_size, H = c.n_head, hd = C / H;
    int t, v, l, i, h, t2;
    ActOff a; act_layout(m, &a);
    double *A = m->acts;
    double *logits = (double *)malloc(sizeof(double) * (long)T * V);
    double *dlogits = (double *)malloc(sizeof(double) * (long)T * V);
    double loss = 0;

    model_forward(m, idx, T, logits);
    for (t = 0; t < T; ++t) {
        double *row = logits + (long)t * V, mx = row[0], denom = 0;
        for (v = 1; v < V; ++v) if (row[v] > mx) mx = row[v];
        for (v = 0; v < V; ++v) denom += exp(row[v] - mx);
        for (v = 0; v < V; ++v) {
            double p = exp(row[v] - mx) / denom;
            dlogits[(long)t * V + v] = (p - (v == targets[t] ? 1.0 : 0.0)) / ((double)T * n_batch);
        }
        loss += -(row[targets[t]] - mx - log(denom));
    }

    /* recompute final LN forward pieces needed for its backward */
    double *xf = A + a.xout + (long)(c.n_layer - 1) * c.block_size * C;
    double *lnf = (double *)malloc(sizeof(double) * (long)T * C);
    double *mu = (double *)malloc(sizeof(double) * T);
    double *rs = (double *)malloc(sizeof(double) * T);
    ln_fwd(lnf, mu, rs, xf, m->params + m->lnf_g, m->params + m->lnf_b, T, C);

    double *dlnf = (double *)malloc(sizeof(double) * (long)T * C);
    double *dx   = (double *)malloc(sizeof(double) * (long)T * C);
    linear_bwd(dlnf, m->grads + m->head_w, m->grads + m->head_b,
               dlogits, lnf, m->params + m->head_w, T, C, V);
    ln_bwd(dx, m->grads + m->lnf_g, m->grads + m->lnf_b,
           dlnf, xf, mu, rs, m->params + m->lnf_g, T, C);

    /* walk blocks backwards; dx holds grad wrt each block's output */
    double *datty = (double *)malloc(sizeof(double) * (long)T * C);
    double *dln   = (double *)malloc(sizeof(double) * (long)T * 4 * C);
    double *dpre  = (double *)malloc(sizeof(double) * (long)T * 4 * C);
    double *dqkv  = (double *)malloc(sizeof(double) * (long)T * 3 * C);
    double *dtmp  = (double *)malloc(sizeof(double) * (long)T * C);

    for (l = c.n_layer - 1; l >= 0; --l) {
        BlockOffsets *b = &m->blk[l];
        long TC = (long)c.block_size * C;
        double *xin  = (l == 0) ? A + a.x0 : A + a.xout + (long)(l - 1) * TC;
        double *ln1o = A + a.ln1_out + l * TC;
        double *qkv  = A + a.qkv + (long)l * c.block_size * 3 * C;
        double *att  = A + a.att + (long)l * H * c.block_size * c.block_size;
        double *atty = A + a.atty + l * TC;
        double *xat  = A + a.xattn + l * TC;
        double *ln2o = A + a.ln2_out + l * TC;
        double *f1p  = A + a.fc1_pre + (long)l * c.block_size * 4 * C;
        double *f1a  = A + a.fc1_act + (long)l * c.block_size * 4 * C;

        /* MLP branch: dx -> fc2 -> gelu -> fc1 -> ln2 -> (+ residual into dxat) */
        linear_bwd(dpre /* d f1a */, m->grads + b->fc2_w, m->grads + b->fc2_b,
                   dx, f1a, m->params + b->fc2_w, T, 4 * C, C);
        for (t = 0; t < T * 4 * C; ++t) dpre[t] *= gelu_grad(f1p[t]);
        linear_bwd(dln /* d ln2o, reuse buffer, only T*C used */,
                   m->grads + b->fc1_w, m->grads + b->fc1_b,
                   dpre, ln2o, m->params + b->fc1_w, T, C, 4 * C);
        ln_bwd(dtmp, m->grads + b->ln2_g, m->grads + b->ln2_b,
               dln, xat, A + a.ln2_mu + (long)l * c.block_size,
               A + a.ln2_rs + (long)l * c.block_size, m->params + b->ln2_g, T, C);
        for (t = 0; t < T * C; ++t) dx[t] += dtmp[t];  /* residual: dxat = dx + dmlp_in */

        /* attention branch: dx(at) -> proj -> attention -> ln1 -> (+ residual) */
        linear_bwd(datty, m->grads + b->proj_w, m->grads + b->proj_b,
                   dx, atty, m->params + b->proj_w, T, C, C);
        memset(dqkv, 0, sizeof(double) * (long)T * 3 * C);
        for (h = 0; h < H; ++h) {
            double scale = 1.0 / sqrt((double)hd);
            for (t = 0; t < T; ++t) {
                const double *arow = att + ((long)h * c.block_size + 0) * 0 /* not used */;
                (void)arow;
                const double *arow_t = att + ((long)h * T + t) * T;
                /* datt and softmax backward */
                double dsum = 0;
                double datt_row[512]; /* T <= block_size <= 512 assumed */
                for (t2 = 0; t2 <= t; ++t2) {
                    const double *vv = qkv + (long)t2 * 3 * C + 2 * C + h * hd;
                    double s = 0;
                    for (i = 0; i < hd; ++i) s += datty[(long)t * C + h * hd + i] * vv[i];
                    datt_row[t2] = s;
                    dsum += s * arow_t[t2];
                }
                for (t2 = 0; t2 <= t; ++t2) {
                    double dscore = arow_t[t2] * (datt_row[t2] - dsum);
                    const double *q = qkv + (long)t * 3 * C + h * hd;
                    const double *k = qkv + (long)t2 * 3 * C + C + h * hd;
                    for (i = 0; i < hd; ++i) {
                        dqkv[(long)t * 3 * C + h * hd + i]      += dscore * scale * k[i];
                        dqkv[(long)t2 * 3 * C + C + h * hd + i] += dscore * scale * q[i];
                    }
                    /* dv += att * datty */
                    for (i = 0; i < hd; ++i)
                        dqkv[(long)t2 * 3 * C + 2 * C + h * hd + i]
                            += arow_t[t2] * datty[(long)t * C + h * hd + i];
                }
            }
        }
        linear_bwd(dtmp /* d ln1o */, m->grads + b->qkv_w, m->grads + b->qkv_b,
                   dqkv, ln1o, m->params + b->qkv_w, T, C, 3 * C);
        {
            double *dxin = datty; /* reuse buffer for d(xin) from ln1 */
            ln_bwd(dxin, m->grads + b->ln1_g, m->grads + b->ln1_b,
                   dtmp, xin, A + a.ln1_mu + (long)l * c.block_size,
                   A + a.ln1_rs + (long)l * c.block_size, m->params + b->ln1_g, T, C);
            for (t = 0; t < T * C; ++t) dx[t] += dxin[t]; /* residual path */
        }
    }

    /* embeddings */
    for (t = 0; t < T; ++t)
        for (i = 0; i < C; ++i) {
            m->grads[m->tok_emb + (long)idx[t] * C + i] += dx[(long)t * C + i];
            m->grads[m->pos_emb + (long)t * C + i]      += dx[(long)t * C + i];
        }

    free(logits); free(dlogits); free(lnf); free(mu); free(rs);
    free(dlnf); free(dx); free(datty); free(dln); free(dpre); free(dqkv); free(dtmp);
    return loss / T;
}

/* ---------------- structure printer & checkpoint ---------------- */

static long block_params(const MiniGPT *m) {
    long C = m->cfg.n_embd;
    return 2*C + (3*C*C + 3*C) + (C*C + C) + 2*C + (4*C*C + 4*C) + (4*C*C + C);
}

void model_print_structure(const MiniGPT *m, FILE *f) {
    GPTConfig c = m->cfg;
    long C = c.n_embd, V = c.vocab_size, Tm = c.block_size;
    int l;
    fprintf(f, "MiniGPT  (total parameters: %ld)\n", m->n_params);
    fprintf(f, "|-- token_embedding      [%ld x %ld]        %ld\n", V, C, V*C);
    fprintf(f, "|-- position_embedding   [%ld x %ld]        %ld\n", Tm, C, Tm*C);
    for (l = 0; l < c.n_layer; ++l) {
        fprintf(f, "|-- block[%d]                              %ld\n", l, block_params(m));
        fprintf(f, "|   |-- ln1              gamma/beta [%ld]   %ld\n", C, 2*C);
        fprintf(f, "|   |-- attn.qkv         [%ld x %ld] + b    %ld\n", 3*C, C, 3*C*C+3*C);
        fprintf(f, "|   |-- attn.proj        [%ld x %ld]  + b    %ld\n", C, C, C*C+C);
        fprintf(f, "|   |-- ln2              gamma/beta [%ld]   %ld\n", C, 2*C);
        fprintf(f, "|   |-- mlp.fc1 (GELU)   [%ld x %ld] + b    %ld\n", 4*C, C, 4*C*C+4*C);
        fprintf(f, "|   `-- mlp.fc2          [%ld x %ld] + b    %ld\n", C, 4*C, 4*C*C+C);
    }
    fprintf(f, "|-- ln_f                 gamma/beta [%ld]   %ld\n", C, 2*C);
    fprintf(f, "`-- head                 [%ld x %ld] + b    %ld\n", V, C, V*C+V);
    fprintf(f, "\nforward:  tokens -> tok_emb + pos_emb -> %d x [ x + Attn(LN1(x)) ; x + MLP(LN2(x)) ] -> LN_f -> head -> logits\n", c.n_layer);
    fprintf(f, "attention: %d heads, head_dim %ld, causal mask, softmax(QK^T/sqrt(d))V\n", c.n_head, C / c.n_head);
}

int model_save(const MiniGPT *m, const CharTokenizer *tok, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const char magic[8] = "MGPTC01\0";
    fwrite(magic, 1, 8, f);
    fwrite(&m->cfg, sizeof(GPTConfig), 1, f);
    fwrite(&tok->vocab_size, sizeof(int), 1, f);
    fwrite(tok->itos, 1, TOK_MAX_VOCAB, f);
    fwrite(&m->n_params, sizeof(long), 1, f);
    fwrite(m->params, sizeof(double), m->n_params, f);
    fclose(f);
    return 0;
}

int model_load(MiniGPT *m, CharTokenizer *tok, const char *path) {
    FILE *f = fopen(path, "rb");
    char magic[8];
    GPTConfig cfg;
    long n;
    int i;
    Rng dummy; rng_seed(&dummy, 1);
    if (!f) return -1;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "MGPTC01", 7) != 0) { fclose(f); return -2; }
    if (fread(&cfg, sizeof(GPTConfig), 1, f) != 1) { fclose(f); return -2; }
    if (fread(&tok->vocab_size, sizeof(int), 1, f) != 1) { fclose(f); return -2; }
    if (fread(tok->itos, 1, TOK_MAX_VOCAB, f) != TOK_MAX_VOCAB) { fclose(f); return -2; }
    for (i = 0; i < TOK_MAX_VOCAB; ++i) tok->stoi[i] = -1;
    for (i = 0; i < tok->vocab_size; ++i) tok->stoi[tok->itos[i]] = i;
    model_init(m, cfg, &dummy);
    if (fread(&n, sizeof(long), 1, f) != 1 || n != m->n_params) { fclose(f); return -3; }
    if (fread(m->params, sizeof(double), n, f) != (size_t)n) { fclose(f); return -3; }
    fclose(f);
    return 0;
}
