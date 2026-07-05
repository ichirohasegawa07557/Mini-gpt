#ifndef MINIGPT_VIZ_H
#define MINIGPT_VIZ_H

#include "minigpt/train.h"

/* SVG writers — replace matplotlib (src/visualize.py). No image libraries. */
int viz_training_curve_svg(const char *path, const HistRow *hist, int n);
int viz_attention_map_svg(const char *path, const double *att, int T,
                          const char *labels /* T chars or NULL */);

#endif
