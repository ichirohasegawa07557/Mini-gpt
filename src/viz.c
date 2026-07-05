#include "minigpt/viz.h"
#include <math.h>

/* matplotlib replacement: hand-written SVG, web-safe fonts + generic fallback */
#define FONT "Arial, Helvetica, sans-serif"

int viz_training_curve_svg(const char *path, const HistRow *hist, int n) {
    FILE *f = fopen(path, "w");
    int i;
    double lo = 1e30, hi = -1e30, x0 = 70, y0 = 40, W = 640, H = 360;
    if (!f || n < 1) { if (f) fclose(f); return -1; }
    for (i = 0; i < n; ++i) {
        if (hist[i].train_loss < lo) lo = hist[i].train_loss;
        if (hist[i].val_loss  < lo) lo = hist[i].val_loss;
        if (hist[i].train_loss > hi) hi = hist[i].train_loss;
        if (hist[i].val_loss  > hi) hi = hist[i].val_loss;
    }
    if (hi - lo < 1e-9) hi = lo + 1.0;
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"760\" height=\"460\">");
    fprintf(f, "<rect x=\"0\" y=\"0\" width=\"760\" height=\"460\" fill=\"#ffffff\"/>");
    fprintf(f, "<text x=\"70\" y=\"28\" font-family=\"%s\" font-size=\"18\" font-weight=\"bold\">Training curve (cross-entropy loss)</text>", FONT);
    fprintf(f, "<rect x=\"%.0f\" y=\"%.0f\" width=\"%.0f\" height=\"%.0f\" fill=\"none\" stroke=\"#888\"/>", x0, y0, W, H);
    for (i = 0; i <= 4; ++i) {
        double v = hi - (hi - lo) * i / 4.0, y = y0 + H * i / 4.0;
        fprintf(f, "<line x1=\"%.0f\" y1=\"%.1f\" x2=\"%.0f\" y2=\"%.1f\" stroke=\"#eee\"/>", x0, y, x0 + W, y);
        fprintf(f, "<text x=\"8\" y=\"%.1f\" font-family=\"%s\" font-size=\"12\">%.3f</text>", y + 4, FONT, v);
    }
    {
        const char *color[2] = {"#1f77b4", "#d62728"};
        int series;
        for (series = 0; series < 2; ++series) {
            fprintf(f, "<polyline fill=\"none\" stroke=\"%s\" stroke-width=\"2\" points=\"", color[series]);
            for (i = 0; i < n; ++i) {
                double vx = n == 1 ? 0.5 : (double)i / (n - 1);
                double v = series == 0 ? hist[i].train_loss : hist[i].val_loss;
                fprintf(f, "%.1f,%.1f ", x0 + vx * W, y0 + (hi - v) / (hi - lo) * H);
            }
            fprintf(f, "\"/>");
        }
    }
    fprintf(f, "<text x=\"%.0f\" y=\"%.0f\" font-family=\"%s\" font-size=\"13\" fill=\"#1f77b4\">train_loss</text>", x0 + W - 150, y0 + 18, FONT);
    fprintf(f, "<text x=\"%.0f\" y=\"%.0f\" font-family=\"%s\" font-size=\"13\" fill=\"#d62728\">val_loss</text>", x0 + W - 150, y0 + 36, FONT);
    fprintf(f, "<text x=\"%.0f\" y=\"%.0f\" font-family=\"%s\" font-size=\"13\">training step (%d..%d)</text>", x0 + W / 2 - 60, y0 + H + 34, FONT, hist[0].step, hist[n-1].step);
    fprintf(f, "</svg>\n");
    fclose(f);
    return 0;
}

int viz_attention_map_svg(const char *path, const double *att, int T, const char *labels) {
    FILE *f = fopen(path, "w");
    int t, t2, cell = T <= 32 ? 18 : 9, x0 = 60, y0 = 60;
    if (!f || T < 1) { if (f) fclose(f); return -1; }
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\">",
            x0 + T * cell + 40, y0 + T * cell + 40);
    fprintf(f, "<rect width=\"100%%\" height=\"100%%\" fill=\"#ffffff\"/>");
    fprintf(f, "<text x=\"%d\" y=\"30\" font-family=\"%s\" font-size=\"16\" font-weight=\"bold\">Attention map — block 0, head 0 (rows attend to columns)</text>", x0, FONT);
    for (t = 0; t < T; ++t) {
        for (t2 = 0; t2 < T; ++t2) {
            double a = att[(long)t * T + t2];
            int shade = 255 - (int)(a * 255.0 + 0.5);
            if (shade < 0) shade = 0;
            if (shade > 255) shade = 255;
            fprintf(f, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"rgb(%d,%d,255)\"/>",
                    x0 + t2 * cell, y0 + t * cell, cell, cell, shade, shade);
        }
        if (labels && cell >= 12) {
            char c = labels[t] == '\n' ? ' ' : labels[t];
            fprintf(f, "<text x=\"%d\" y=\"%d\" font-family=\"%s\" font-size=\"11\">%c</text>",
                    x0 - 16, y0 + t * cell + cell - 4, FONT, c);
            fprintf(f, "<text x=\"%d\" y=\"%d\" font-family=\"%s\" font-size=\"11\">%c</text>",
                    x0 + t * cell + 4, y0 - 8, FONT, c);
        }
    }
    fprintf(f, "</svg>\n");
    fclose(f);
    return 0;
}
