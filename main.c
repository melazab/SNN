/* SNN transformer architecture in pure C. */
/* Based on Spiking Manifesto (Izhikevich 2025) */
/* The code strives for simplicity, not efficiency. */
/* Eugene Izhikevich, October 2025 */
/* Forked by Mohamed Elazab, November 2026 */

/* ============================================================================
 * 0. WHAT THIS PROGRAM DOES
 * ----------------------------------------------------------------------------
 * It reads a big text file and learns to predict the next character (byte)
 * given the previous 32 characters. "Learning" means: repeatedly show it a
 * 33-character snippet, let it guess characters 2..33 from the characters
 * before each of them, measure how wrong the guesses were, and nudge its
 * internal numbers so that next time it is a bit less wrong. Every 10,000
 * snippets it measures its accuracy on snippets it never trains on and prints
 * some text it generates by itself. That is exactly what a language model
 * (like GPT) is, only tiny and character-level.
 *
 * ----------------------------------------------------------------------------
 * 1. VOCABULARY
 * ----------------------------------------------------------------------------
 * token           One unit of input. Here a token = one byte (0..255) of the
 *                 text file. VOCAB_SIZE = 256 is the number of possible
 *                 tokens.
 * context         The window of previous tokens the model sees:
 *                 CONTEXT_SIZE = 32.
 * vector          Just an array of floats. "A vector in R^n" = float[n].
 * embedding       A vector that represents a token. Token 'a' -> some
 *                 float[32]. The table that does this is
 *                 Token_embedder[256][32].
 * layer           One processing stage; the model stacks NUM_LAYERS = 6.
 * logits          The model's raw output scores, one per possible next token
 *                 (256 of them). Larger = "more likely". Not yet
 *                 probabilities.
 * softmax         Turns logits into probabilities:
 *                     p_k = exp(s_k) / sum_m exp(s_m)
 *                 All p_k are positive and add up to 1.
 * loss            A single number measuring how wrong a prediction is. Here
 *                 it is "cross-entropy": loss = -log(p_correct). If the model
 *                 gave the correct next character probability 1, loss = 0;
 *                 probability 0.5, loss = 0.69; probability 0.01,
 *                 loss = 4.6. Lower is better.
 * BPC             Bits Per Character = loss / ln(2). The paper reports BPC
 *                 (Table II). This code logs loss in natural-log units
 *                 ("nats"), so divide the numbers in loss.csv by 0.693 to
 *                 compare with the paper.
 * perplexity      exp(loss). "The model is as unsure as if it were choosing
 *                 uniformly among this many characters." The paper stops
 *                 training at perplexity < 2.0, i.e. loss < 0.693 nats,
 *                 i.e. BPC < 1.0 (Sect. V).
 * gradient        For every adjustable number w in the model, dLoss/dw: "if
 *                 I increase w a tiny bit, how much does the loss go up?" To
 *                 reduce the loss, move w the OPPOSITE way:
 *                     w -= learning_rate * dLoss/dw
 *                 This is "gradient descent" (here plain "SGD": stochastic
 *                 gradient descent, stochastic because each step uses one
 *                 random snippet).
 * learning rate   The step size in the update above (the variable
 *                 `learning_rate`). The paper calls it epsilon (Fig. 10,
 *                 line 29).
 * backpropagation The chain rule of calculus applied layer by layer from the
 *                 output back to the input, to get every gradient
 *                 efficiently. "Forward pass" = compute the prediction;
 *                 "backward pass" = compute gradients and update.
 *                 See Sect. III-B, Eq. (6)-(8), Fig. 10.
 * residual (skip) Instead of output = f(input), use
 *   connection        output = input + f(input).
 *                 In this code that's why everything is written with "+="
 *                 into m->z. It makes deep stacks much easier to train.
 *                 Eq. (10), Fig. 9.
 * parameter       An adjustable number that learning changes. In ordinary
 *   (weight)      networks these are matrix entries; here they are the
 *                 entries of the look-up tables, which the paper calls
 *                 "synaptic values" s_ijk (Sect. II).
 * attention       The mechanism that lets position i in the context use
 *                 information from earlier positions j < i. A standard
 *                 transformer uses dot products + softmax (Eq. 13). Here it
 *                 is a look-up table indexed by the PAIR (z_i, z_j) plus
 *                 their distance (Eq. 14, 15).
 * head            One independent copy of the attention mechanism;
 *                 NUM_HEADS = 4 per layer, all added together.
 * FFN             "Feed-forward network": a per-position processing step
 *                 after attention. Here it is a single look-up
 *                 transformation (Eq. 12).
 *
 * ----------------------------------------------------------------------------
 * 2. THE CORE IDEA OF THE SPIKING MANIFESTO
 * ----------------------------------------------------------------------------
 * Paper: a vector x in R^n is read as "spike latencies": x[k] = WHEN neuron k
 * fired. What matters is only the ORDER of spikes between chosen pairs of
 * "anchor" neurons: did neuron a fire before neuron b? (Sect. II, Figs. 5, 6.)
 *
 * One LOOK-UP TABLE (paper: "table i"; code: S[i] inside struct LUT) does:
 *
 *   ASSIGN       pick n_c random pairs (a_r, b_r) once     -> build_LUT()
 *   COMPARE      u_r = x[a_r] - x[b_r] for each pair       -> cache_index()
 *   CONCATENATE  bit_r = (u_r > 0); j = bits as an integer -> cache_index()
 *   SELECT       output += row j of the table              -> LUT_forward()
 *
 * A "LUT" in the paper's sense (the "look-up transformation S_x", Eq. 3) is
 * n_t such tables whose selected rows are summed:
 *
 *     y_k = sum_{i=1..n_t} s_{i, H_i(x), k}
 *
 * NAME MAP
 *   paper symbol               code
 *   ------------               ----
 *   n                          EMBEDDING_DIM (= 32), length of x, z, y
 *   n_t                        N_T (= 16), number of tables per LUT
 *   n_c                        N_C (= 6), comparisons per table
 *   2^{n_c}                    rows per ordinary table (= 64)
 *   p                          POSITIONAL_DIM (= 4), length of a PE vector
 *   n_inp                      CONTEXT_SIZE (= 32)
 *   N                          NUM_LAYERS (= 6)
 *   h                          NUM_HEADS (= 4)
 *   a_i, b_i                   lut->anchors[i].a[], lut->anchors[i].b[]
 *   s_ijk                      lut->S[i][j * y_dim + k]
 *                              (table i, row j, column k)
 *   S_ix                       the row starting at lut->S[i] + j * y_dim,
 *                              with j = H_i(x)
 *   H_i(x) = j                 cache->j[i]
 *   u_i (u_ir with min |u_ir|) cache->u_min[i]; its r is cache->r_min[i]
 *   j-bar, x-bar (Figs. 7, 14) jbar (j with bit r_min flipped)
 *   U(u), U'(u) (Eq. 4, 5)     macros U(x), Up(x)
 *   g_i (Eq. 7)                local variable gi in BACKWARD_UPDATE
 *   v^l (Fig. 10)              x_grad / y_grad arrays (dLoss/dx)
 *   epsilon                    learning_rate
 *   z_i (attention input)      m->z[i]
 *   PE_{i-j}                   head->Positional_encoding[i-j]
 *   V (attention LUT, Eq. 15)  head->V
 *   V-index cache (IV-D)       head->V_cache[] and head->PE_cache[]
 *   E (embedder)               Token_embedder
 *   U_h (unembedder)           unembedder
 *
 * ----------------------------------------------------------------------------
 * 3. THE WHOLE MODEL IN PSEUDOCODE (Fig. 11b, "SNN transformer")
 * ----------------------------------------------------------------------------
 *   z[pos] = Token_embedder[token[pos]]                    for pos = 0..31
 *   repeat for each of 6 layers:
 *       for each of 4 heads, for each pos:                 (attention, Eq. 15)
 *           z[pos] += sum over pos1 < pos of
 *                         V[z[pos], z[pos1], PE[pos - pos1]]
 *       for each pos:                                      (FFN, Eq. 12)
 *           z[pos] += FFN[z[pos]]
 *   logits[pos] = unembedder[z[pos]]
 *   prob[pos]   = softmax(logits[pos])          -> guess for token[pos + 1]
 *
 * where "LUT[x]" means "compute indices from x by anchor comparisons, then add
 * up the selected rows". There is NO multiplication anywhere in this forward
 * pass: only comparisons, bit-twiddling, memory reads and additions (plus
 * exp() in the final softmax). That is the paper's efficiency argument
 * (Sect. II; Table IV).
 *
 * ----------------------------------------------------------------------------
 * 4. HOW LEARNING WORKS (Sect. III), the part most unlike normal ML
 * ----------------------------------------------------------------------------
 * Two kinds of things need gradients:
 *
 *  (a) The table entries s_ijk. Easy: output y = sum of selected rows, so to
 *      move y in the direction -dLoss/dy, subtract learning_rate * dLoss/dy
 *      from each selected row. Rows that weren't selected don't change.
 *      (Fig. 10, line 29.)
 *
 *  (b) The INPUT x of each table, so the error can flow to earlier layers.
 *      Hard: j = H_i(x) is a step function (integer-valued); its true
 *      derivative is 0 almost everywhere. The paper's trick ("surrogate
 *      gradient", Eq. 4):
 *        - find the anchor pair whose difference u is closest to 0: that's
 *          the comparison a tiny change in x could flip;
 *        - pretend the output is a blend:
 *              S_x + U(u) * (S_xbar - S_x)
 *          where x-bar is "x with that one comparison flipped" and
 *          U(u) = 0.5/(1+|u|) is 0.5 at u = 0 and fades to 0 for large |u|;
 *        - differentiate THAT smooth formula (Eq. 5, 8).
 *      Intuition (Sect. III-B): g_i compares the flipped row with the current
 *      row along the error direction: "would flipping this comparison have
 *      helped?". If yes, x[a] and x[b] get pushed toward each other (toward
 *      flipping); if no, they get pushed apart (making the current pattern
 *      more certain).
 *
 *  The blend is used ONLY to compute gradients; the forward pass always uses
 *  the hard lookup (end of Sect. III-A).
 *
 * ----------------------------------------------------------------------------
 * 5. WHERE THIS FILE DIFFERS FROM, OR GOES BEYOND, THE PAPER
 * ----------------------------------------------------------------------------
 *  - README "Deviations" 1-4 (per-table positional encoders; PE compared to 0
 *    instead of anchors; shared Q/K anchors; frozen Token_embedder).
 *  - The attention backward pass (concatenated_LUT_backward) can update BOTH
 *    a Q/K anchor pair AND a PE bit in the same table; see the note there.
 *  - The code calls its schedule "Adam"; it is the learning-RATE schedule
 *    from the original Transformer paper, applied to plain SGD. No Adam
 *    optimizer is used.
 *  - Validation loss is measured on the LAST position only; training uses
 *    all 32.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
// fcntl/unistd/mman are not actually used below.
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// saves validation loss values during training
#define FILE_NAME "loss.csv"

/* ---------------- HYPERPARAMETERS ----------------
 * "Hyperparameters" are sizes and settings chosen by hand (learning doesn't
 * change them). Compare with Table III, column "SNN (green curve)", which
 * lists exactly these values.
 *
 *   CONTEXT_SIZE    n_inp: how many previous characters the model sees
 *   VOCAB_SIZE      one token per possible byte value
 *   EMBEDDING_DIM   n: length of every internal vector ("number of neurons"
 *                   in a spiking pattern)
 *   POSITIONAL_DIM  p: length of each positional-encoder vector -> 4 extra
 *                   index bits in attention
 *   NUM_LAYERS      N
 *   NUM_HEADS       h
 *   N_T             n_t: number of look-up tables inside every LUT
 *   N_C             n_c: comparisons per table -> 2^6 = 64 rows per table
 */

#define CONTEXT_SIZE   32
#define VOCAB_SIZE     256
#define EMBEDDING_DIM  32
#define POSITIONAL_DIM 4
#define NUM_LAYERS     6
#define NUM_HEADS      4

#define N_T 16
#define N_C 6

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
// zero has "minus" sign
// consistent with cache_index(): u > 0 -> bit 1, u <= 0 -> bit 0.
#define sign(x) (x > 0 ? 1 : -1)

/* U(u) is the "uncertainty" function of Fig. 8 (left): U(0) = 0.5,
 * and U -> 0 as |u| grows. It says how much to blend the current row with the
 * flipped row in the surrogate Eq. (4). Only its derivative U'(u) = Up(u) is
 * ever needed (Eq. 5, 8):
 *
 *     d/du [ 0.5 / (1+|u|) ] = -0.5 * sign(u) / (1+|u|)^2
 *
 * For u > 0, U' < 0 (making u bigger makes a flip less plausible); for u < 0,
 * U' > 0. |U'| is largest near u = 0, so learning concentrates on comparisons
 * that are "on the fence" -- the paper's "plasticity window" (last paragraph
 * of Sect. III-A). */
#define U(x)  (0.5 / (1 + fabs(x))) // not used; here for reference
#define Up(x) (-0.5 * sign(x) / (1 + fabs(x)) / (1 + fabs(x)))

/* LEARNING-RATE SCHEDULE.
 * Despite the comment below, this is not the Adam *optimizer*; it is the
 * schedule from "Attention Is All You Need" (Vaswani et al. 2017), which the
 * paper reuses (Fig. 13 caption): grow linearly for 4000 steps ("warm-up"),
 * then decay like 1/sqrt(t).
 *
 *     t <  4000:  lr = t / 4000^1.5   (rises from 0 to ~0.0158)
 *     t >= 4000:  lr = 1 / sqrt(1+t)  (0.0158 at t = 4000;
 *                                      0.001 at t = 1,000,000)
 *
 * `t` is the step counter in main(); the macro is expanded there, where `t`
 * exists. 4000.0f (a float, not an int) is the fix from the latest upstream
 * commit: with integer division, t/4000 was 0 for t < 4000, so the model
 * learned nothing at first. */
// Adam learning rate scheduler
#define LEARNING_RATE (MIN(1 / sqrt(1 + t), t / (4000.0f) / sqrt(4000)))
// global "epsilon"; set once per training step in main()
float learning_rate;
// number of held-out snippets used to measure validation loss
#define TESTING_LENGTH 10000

// ----------------------------------------------------------------------------
// Spiking model

/* ---------------- DATA STRUCTURES ----------------
 *
 * Anchors: the n_c anchor pairs of ONE table (paper: a_i and b_i, Sect. II).
 * a[r] and b[r] are indices into the input vector x (0..EMBEDDING_DIM-1).
 * Comparison r of this table asks "is x[a[r]] > x[b[r]]?", which in spike
 * language is "did neuron b[r] fire before neuron a[r]?" (or its mirror
 * image; only consistency matters). Chosen randomly once and never learned.
 * Sect. VIII-I discusses learning them ("structural plasticity") as future
 * work. */
typedef struct {
    int a[N_C];
    int b[N_C];
} Anchors;

/* LUT = the paper's "look-up transformation" S_x (Eq. 3): N_T
 * tables, summed.
 *
 * Memory layout of one table S[i]: a flat float array of
 * (number_of_rows * y_dim) entries:
 *
 *       row 0 : s_{i,0,0}  s_{i,0,1}  ...  s_{i,0,y_dim-1}
 *       row 1 : s_{i,1,0}  ...
 *       ...
 *       row j starts at S[i] + j*y_dim
 *
 * number_of_rows = 2^(number of index bits). For an ordinary LUT that is
 * 2^N_C = 64. For the attention LUT the index has N_C (query) + N_C (key) +
 * POSITIONAL_DIM bits = 16 bits = 65,536 rows (see build_Model). A row is one
 * "synaptic vector": what firing pattern j sends to each of the y_dim output
 * neurons.
 *
 * Paper's view (Figs. 6, 16): each row is a "detector neuron" tuned to one
 * ordering of the anchor spikes; exactly one detector per table fires, and
 * only its synapses are used. The rest of the (huge) table is idle for this
 * input. */
typedef struct {
    int y_dim; // dimension of the output y
    // synaptic values (table_size, y_dim), i.e., (,j,k) in the paper
    float *S[N_T];
    Anchors anchors[N_T];
} LUT;

/* LUTcache: what the forward pass remembers about ONE input vector x
 * for ONE LUT, so the backward pass doesn't need x itself (paper: "index
 * caching", Fig. 10 lines 13 and 24; Sect. IV-A). Per table i:
 *
 *   j[i]      H_i(x), the row that was selected (Eq. 1)
 *   r_min[i]  which of the N_C comparisons had the smallest |u|
 *             (the one to "flip")
 *   u_min[i]  that u value itself (signed), needed for U'(u)
 *
 * These three numbers per table are everything the learning rule (Eq. 7, 8)
 * needs. */
typedef struct {
    int r_min[N_T];
    float u_min[N_T];
    int j[N_T]; // j = H_i(x)
} LUTcache;

/* One attention head (Sect. IV-C, IV-D; Fig. 12).
 *
 * V             The attention LUT of Eq. (15). Its input is conceptually
 *               the concatenated vector [z_i, z_j, PE_{i-j}] (Eq. 14), but
 *               that vector is never built: the three index pieces are glued
 *               together (CONCATENATE).
 * V_cache[pos]  The "V-index cache" (Sect. IV-D). For every position pos,
 *               the bits from comparing anchors inside z[pos]. Computed ONCE
 *               per position and reused for every pair (pos, pos1), both as
 *               the "query" half (when pos is the later position) and as the
 *               "key" half (when it's the earlier one). This reuse is README
 *               deviation 3.
 * Positional_encoding[d][i][:]
 *               The learnable vector PE_d for relative distance d,
 *               separately for each table i (README deviation 1). Its signs
 *               (> 0 or not) give POSITIONAL_DIM index bits (README
 *               deviation 2; last formula of Sect. VIII-D). Only d = 1..31 is
 *               used: a token never attends to itself (Sect. IV-C: including
 *               j = i hurt).
 * PE_cache[d]   Index bits (and closest-to-zero entry) of PE_d, per table.
 */
typedef struct {
    LUT V;
    LUTcache V_cache[CONTEXT_SIZE];

    // position -> embedding
    float Positional_encoding[CONTEXT_SIZE][N_T][POSITIONAL_DIM];
    LUTcache PE_cache[CONTEXT_SIZE];

} AttentionHead;

/* The whole model (Fig. 11b, right column).
 *
 * Token_embedder   The embedder table E (Sect. IV-C, Table I). Row t =
 *                  vector for byte t. Random and frozen in this code (README
 *                  deviation 4).
 * tokens           The 33 bytes of the current snippet. tokens[0..31] are
 *                  the input; tokens[pos+1] is the "correct answer" for
 *                  position pos.
 * z                The "residual stream". z[pos] is the current 32-float
 *                  vector (spiking pattern / latency vector) at context
 *                  position pos. Every block READS z and ADDS its output back
 *                  into z (Eq. 10, 12, 15).
 * FFN[l]           Feed-forward LUT of layer l (Eq. 12), 32 -> 32.
 * FFN_cache        Its index caches, per layer and position.
 * head[l][h]       Attention heads.
 * unembedder       LUT 32 -> 256 producing logits (paper's U_h, Sect. IV-C).
 * unembedder_vars  Its index caches (despite the name, the same type as the
 *                  other caches).
 * output           Logits per position. After softmax and subtracting 1 at
 *                  the correct token, it is REUSED to hold the gradient
 *                  dLoss/dlogits (see model_training_step).
 */
typedef struct {
    float Token_embedder[VOCAB_SIZE][EMBEDDING_DIM]; // token -> embedding
    int tokens[CONTEXT_SIZE + 1];                    // currrent training sample

    // resnet connections: each layer adds to z
    float z[CONTEXT_SIZE][EMBEDDING_DIM];

    LUT FFN[NUM_LAYERS];
    LUTcache FFN_cache[NUM_LAYERS][CONTEXT_SIZE];

    AttentionHead head[NUM_LAYERS][NUM_HEADS];

    LUT unembedder;
    LUTcache unembedder_vars[CONTEXT_SIZE];

    // the output becomes the error gradient in the backward pass
    float output[CONTEXT_SIZE][VOCAB_SIZE];
} Model;

/* softmax: turn arbitrary scores into probabilities, in place.
 *
 *     p_k = exp(x_k / T) / sum_m exp(x_m / T)
 *
 * Subtracting the max first doesn't change the result (it cancels in the
 * ratio) but keeps exp() from overflowing. `temperature` T: T = 1 for
 * training and validation; T < 1 (0.4 in model_inference) sharpens the
 * distribution, so generated text is less random. This is the only place
 * exp() is used: at the very output (Fig. 11b, top box). */
void softmax(float *x, int size, float temperature) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf((x[i] - max_val) / temperature);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Dot product. Never called in this file (no matrix multiplication
// is needed).
float vector_multiply(float *vector1, float *vector2, int size) {
    float result = 0;
    for (int i = 0; i < size; i++) {
        result += vector1[i] * vector2[i];
    }
    return result;
}

// Fill with uniform random numbers in [-scale, +scale].
void random_vector(float *vector, int size, float scale) {
    for (int i = 0; i < size; i++) {
        vector[i] = scale * 2 * ((float)rand() / RAND_MAX - 0.5);
    }
}

/* Draw one index at random according to `probabilities`
 * (inverse-CDF method): walk along the running sum until it exceeds a
 * uniform random number in [0,1). Used only for text generation. */
int sample(float *probabilities, int n) {
    // sample index from probabilities (they must sum to 1!)
    float coin = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

void fill_vector_with_random_intergers(int *vector, int N, int Max_value) {

    for (int i = 0; i < N; i++) {
        vector[i] = rand() % Max_value;
    }
}

// Ensures b[r] != a[r] (Fig. 10 line 2: "random pairs a != b").
// Comparing a neuron with itself would always give u = 0, a useless constant
// bit. Note: nothing prevents the same pair from appearing twice, in one
// table or across tables.
void fill_vector_with_random_intergers_different_from_vector2(int *vector,
                                                              int *vector2,
                                                              int N,
                                                              int Max_value) {

    for (int i = 0; i < N; i++) {
        do {
            vector[i] = rand() % Max_value;
        } while (vector2[i] == vector[i]);
    }
}

/* build_LUT: the ASSIGN step (Sect. II) plus memory allocation.
 *
 *   total_n_c  number of index bits for this LUT -> each table has
 *              2^total_n_c rows. N_C for FFN/unembedder;
 *              N_C + N_C + POSITIONAL_DIM for attention.
 *   y_dim      output vector length (= row length).
 *
 * Anchors are drawn from 0..EMBEDDING_DIM-1 because every LUT here reads a z
 * vector. For the attention LUT only N_C anchor pairs per table are drawn
 * (not 2*N_C): the same pairs are applied to z_i and to z_j (README deviation
 * 3). The PE bits need no anchors at all (deviation 2).
 *
 * All synaptic values start at 0 (Sect. V: "initialize all synapses to 0"),
 * so at first every LUT outputs the zero vector.
 *
 * Memory: an attention table is 2^16 rows * 32 floats * 4 bytes = 8 MiB;
 * times 16 tables x 4 heads x 6 layers = 3 GiB (Table III: 806M parameters).
 * The memset after calloc is redundant (calloc already zeroes) and forces the
 * OS to commit all 3 GiB up front. */
void build_LUT(LUT *lut, int total_n_c, int y_dim) {

    lut->y_dim = y_dim;
    for (int i = 0; i < N_T; i++) {
        fill_vector_with_random_intergers(lut->anchors[i].a, N_C,
                                          EMBEDDING_DIM);
        fill_vector_with_random_intergers_different_from_vector2(
            lut->anchors[i].b, lut->anchors[i].a, N_C, EMBEDDING_DIM);
        // 1 << k == 2^k
        lut->S[i] = (float *)calloc((1 << total_n_c) * y_dim, sizeof(float));
        memset(lut->S[i], 0, (1 << total_n_c) * y_dim * sizeof(float));
    }
}

void free_LUT(LUT *lut) {

    for (int i = 0; i < N_T; i++) {
        free(lut->S[i]);
    }
}

/* cache_index: COMPARE + CONCATENATE (Eq. 1; Fig. 10 lines 8-13).
 * For every table i:
 *
 *     u_r   = x[a_r] - x[b_r]   ("latency difference of anchor pair r")
 *     bit_r = (u_r > 0)
 *     j     = bit_0 + 2*bit_1 + 4*bit_2 + ...
 *                               ("concat": bit r goes to position r)
 *
 * and record which r had the smallest |u_r| (needed only for learning;
 * Fig. 7).
 *
 * Worked example in the style of Fig. 6 (n_c = 3): if x1 > x2, x2 < x3,
 * x3 > x5, the bits are (1, 0, 1) -> j = 1 + 0 + 4 = 5. (The figure writes
 * bits most-significant-first; the code puts comparison r at bit r.
 * Different order, same idea, used consistently.)
 *
 * This function never looks at the table contents S, only at x and the
 * anchors. In LSH (locality-sensitive hashing) language (Sect. VI, Fig. 14),
 * j is the "hash bucket" of x: nearby x's usually land in the same bucket. */
void cache_index(LUT *lut, LUTcache *cache, float *x) {

    for (int i = 0; i < N_T; i++) {
        cache->j[i] = 0;
        cache->u_min[i] = INFINITY;

        for (int r = 0; r < N_C; r++) { // loop over all comparisons per table
            float u = x[lut->anchors[i].a[r]] - x[lut->anchors[i].b[r]];
            if (u > 0) {
                cache->j[i] |= (1 << r); // concatenation: set bit r
            }
            if (fabs(u) < fabs(cache->u_min[i])) {
                cache->r_min[i] = r;
                cache->u_min[i] = u;
            }
        }
    }
}

/* cache_PE_index: same as cache_index but for a positional-encoder
 * vector, and WITHOUT anchors: bit r is simply (PE[i][r] > 0). This is the
 * hash
 *
 *     j = concat(x_{a1} > 0, ..., x_{a_nc} > 0)
 *
 * at the end of Sect. VIII-D, which the paper says is used "for the
 * positional encoders PE in R^p when p is small". The argument `u` is
 * head->Positional_encoding[distance], shape [N_T][POSITIONAL_DIM]: a
 * separate small vector per table. */
void cache_PE_index(LUTcache *cache, float u[N_T][POSITIONAL_DIM]) {

    for (int i = 0; i < N_T; i++) {
        cache->j[i] = 0;
        cache->u_min[i] = INFINITY;

        for (int r = 0; r < POSITIONAL_DIM; r++) {
            // No anchor neurons; just a comparison with 0; works best for
            // small POSITIONAL_DIM.
            if (u[i][r] > 0) {
                cache->j[i] |= (1 << r);
            }
            if (fabs(u[i][r]) < fabs(cache->u_min[i])) {
                cache->r_min[i] = r;
                cache->u_min[i] = u[i][r];
            }
        }
    }
}

/* LUT_forward: SELECT + sum (Eq. 2/3; Fig. 10 line 14).
 *
 *     y += sum_i  S_i[row j_i]
 *
 * Note the "+=": the caller passes the vector to add into. For the FFN that
 * is z itself (residual connection, Eq. 12); for the unembedder, the zeroed
 * output array. Cost: N_T row reads and N_T * y_dim additions. No
 * multiplications.
 *
 * Paper (Sect. II): "synaptic transmission" -- the fired detector neuron j of
 * each table delivers its synaptic values to all output neurons k. */
void LUT_forward(LUT *lut, LUTcache *cache, float *y) {

    // main loop in the forward pass
    for (int i = 0; i < N_T; i++) {
        for (int k = 0; k < lut->y_dim; k++) {
            y[k] += lut->S[i][cache->j[i] * lut->y_dim + k];
        }
    }
}

/* BACKWARD_UPDATE: the surrogate-gradient rule for the INPUT of
 * table i (Eq. 7 and 8; Fig. 10 lines 26-28). It is a macro (textual
 * copy-paste) rather than a function so it can use the local variables i, j,
 * jbar, lut and y_gradient of whichever function it's pasted into. Here j and
 * jbar are already multiplied by y_dim, i.e. they are offsets to the start of
 * each row.
 *
 *   y_gradient = dLoss/dy for this LUT's output y
 *                                    (paper: dL/dx^{l+1}, v^{l+1})
 *   gi  = y_gradient . (S_i[jbar] - S_i[j])
 *                                    (Eq. 7: the "alignment" g_i)
 *   v   = gi * U'(u_min)             (Eq. 8)
 *   dLoss/dx[a] += v,  dLoss/dx[b] -= v
 *                                    (because u = x[a] - x[b])
 *
 * Where this comes from. The surrogate output (Eq. 4) is
 *
 *     y = S_j + U(u) * (S_jbar - S_j)
 *
 * so by the chain rule
 *
 *     dLoss/du = (dLoss/dy) . (S_jbar - S_j) * U'(u) = gi * U'(u) = v.
 *
 * What it means (sign check). Gradient descent changes u by -lr * v. Take
 * u > 0, so U'(u) < 0:
 *
 *   gi < 0 : the flipped row S_jbar lies "downhill" (it would lower the
 *            loss). v > 0, so u decreases toward 0, toward flipping the
 *            comparison.
 *   gi > 0 : the flipped row would raise the loss. v < 0, so u grows away
 *            from 0: the current spike order is reinforced.
 *
 * (u < 0 is the mirror image.) Earlier layers receive this as dLoss/dx and
 * adjust their own tables so that x moves this way.
 *
 * Only ONE comparison per table (the one closest to flipping) gets any
 * gradient, split equally and oppositely between its two anchor neurons, so
 * the total gradient into x sums to zero (paper: "self-regularization", end
 * of Sect. III-B). The x-gradients are ADDED (+=) because many tables -- and,
 * through the residual connections, many layers -- contribute to the same x.
 */
#define BACKWARD_UPDATE(cache, gradient)                                       \
    do {                                                                       \
        float gi = 0;                                                          \
        for (int k = 0; k < lut->y_dim; k++) {                                 \
            gi += y_gradient[k] * (lut->S[i][jbar + k] - lut->S[i][j + k]);    \
        }                                                                      \
        float v = gi * Up(cache->u_min[i]);                                    \
        gradient[lut->anchors[i].a[cache->r_min[i]]] += v;                     \
        gradient[lut->anchors[i].b[cache->r_min[i]]] -= v;                     \
    } while (0)
//

/* LUT_backward: learning for one ordinary LUT on one input
 * (Fig. 10 lines 23-30).
 *
 *   x_gradient  ACCUMULATES dLoss/dx for the input x (for the previous layer)
 *   y_gradient  dLoss/dy, given by the layer above
 *
 * Per table:
 *   1. jbar = j with bit r_min flipped (XOR with 1<<r_min toggles exactly
 *      that bit). This is the "nearest other bucket" x-bar of Fig. 14.
 *   2. Send error to the input via the surrogate rule (BACKWARD_UPDATE).
 *      This must happen BEFORE step 3, because it reads S[j], which step 3
 *      changes.
 *   3. Learn the table: the selected row j moves by
 *      -learning_rate * dLoss/dy. Why this is the correct gradient:
 *      y = ... + S_i[j] + ..., so dy_k/dS_i[j][k] = 1 and
 *      dLoss/dS_i[j][k] = dLoss/dy_k. Only row j was used, so only row j
 *      changes. (Fig. 14 caption: "Learning is just a nudge of S_j along
 *      vector v".)
 *
 * The update is applied immediately ("online"), not saved up and applied
 * later. */
void LUT_backward(LUT *lut, LUTcache *cache, float *x_gradient,
                  float *y_gradient) {

    for (int i = 0; i < N_T; i++) {
        int j = cache->j[i] * lut->y_dim;
        // flip the bit number r_min
        int jbar = (cache->j[i] ^ (1 << cache->r_min[i])) * lut->y_dim;

        BACKWARD_UPDATE(cache, x_gradient);

        for (int k = 0; k < lut->y_dim; k++) {
            lut->S[i][j + k] -= learning_rate * y_gradient[k];
        }
    }
}

/* CONCATENATE builds the 16-bit row number of the attention LUT from
 * three cached pieces (V-index cache, Sect. IV-D) and converts it to a row
 * offset (* y_dim):
 *
 *      bit:  15 ........ 10 | 9 ......... 4 | 3 ...... 0
 *            Q: 6 bits from   K: 6 bits from  PE: 4 bits from
 *            z[pos]           z[pos1]         PE[pos-pos1]
 *
 * This is the integer you would get by computing H on the concatenated
 * vector [z_i, z_j, PE_{i-j}] of Eq. (14), without ever building that vector.
 * Paper (end of Sect. IV-D): computing z_i's index ~ "query", z_j's index ~
 * "key", and gluing them and looking up a row ~ "Q.K^T, then times V" in a
 * normal transformer. */
#define CONCATENATE(Q, P, PE)                                                  \
    ((((Q) << (N_C + POSITIONAL_DIM)) | ((P) << POSITIONAL_DIM) | (PE)) *      \
     lut->y_dim)

/* Forward for ONE (query position, key position) pair: one term of
 * Eq. (15)
 *
 *     y += V[z_i, z_j, PE_{i-j}]     (summed over the N_T tables)
 *
 * cacheQ = index cache of the later position i, cacheK = of the earlier
 * position j, cachePE = of their distance. The caller loops over all pairs.
 */
void concatenated_LUT_forward(LUT *lut, LUTcache *cacheQ, LUTcache *cacheK,
                              LUTcache *cachePE, float *y) {

    for (int i = 0; i < N_T; i++) {
        int j = CONCATENATE(cacheQ->j[i], cacheK->j[i], cachePE->j[i]);
        for (int k = 0; k < lut->y_dim; k++) {
            y[k] += lut->S[i][j + k];
        }
    }
}

/* Backward for ONE attention pair. Same three steps as LUT_backward,
 * but the index has three parts, and the "closest to flipping" comparison
 * could be in any of them.
 *
 *  (1) Q vs K: whichever of the query's and the key's closest comparisons is
 *      closer to 0 is flipped; its gradient goes to anchors in z[pos]
 *      (x_gradientQ) or in z[pos1] (x_gradientK).
 *  (2) PE: IF the PE's closest entry is closer to 0 than both, that PE bit is
 *      flipped too and its gradient goes into PE_grad. There are no anchors,
 *      so u = PE[r] itself and the whole v goes to that one entry (no "-v"
 *      partner).
 *      NOTE: (1) still ran, so in that case TWO bits of this table get a
 *      gradient. The paper's general rule (one minimal pair per table) would
 *      do only (2). The paper doesn't spell out the attention backward pass,
 *      so this detail is open.
 *  (3) The selected row learns: S[j] -= lr * dLoss/dy.
 *
 * Many pairs can select the same row, and (3) is applied immediately, so
 * later pairs in the loop see rows already updated by earlier pairs in the
 * same step. */
void concatenated_LUT_backward(LUT *lut, LUTcache *cacheQ, LUTcache *cacheK,
                               LUTcache *cachePE, float *x_gradientQ,
                               float *x_gradientK,
                               float PE_grad[N_T][POSITIONAL_DIM],
                               float *y_gradient) {

    for (int i = 0; i < N_T; i++) {

        int j = CONCATENATE(cacheQ->j[i], cacheK->j[i], cachePE->j[i]);

        if (fabs(cacheQ->u_min[i]) < fabs(cacheK->u_min[i])) {
            // only jQ is flipped
            int jbar = CONCATENATE(cacheQ->j[i] ^ (1 << cacheQ->r_min[i]),
                                   cacheK->j[i], cachePE->j[i]);
            BACKWARD_UPDATE(cacheQ, x_gradientQ);
        } else {
            // only jP is flipped
            int jbar = CONCATENATE(cacheQ->j[i],
                                   cacheK->j[i] ^ (1 << cacheK->r_min[i]),
                                   cachePE->j[i]);
            BACKWARD_UPDATE(cacheK, x_gradientK);
        }

        if (fabs(cachePE->u_min[i]) < fabs(cacheQ->u_min[i]) &&
            fabs(cachePE->u_min[i]) < fabs(cacheK->u_min[i])) {
            // only jPE is flipped
            int jbarPE = CONCATENATE(cacheQ->j[i], cacheK->j[i],
                                     cachePE->j[i] ^ (1 << cachePE->r_min[i]));
            float giPE = 0;
            for (int k = 0; k < lut->y_dim; k++) {
                giPE +=
                    y_gradient[k] * (lut->S[i][jbarPE + k] - lut->S[i][j + k]);
            }
            float deltaPE = giPE * Up(cachePE->u_min[i]);
            // No anchor neurons; just a comparison with 0.
            PE_grad[i][cachePE->r_min[i]] += deltaPE;
        }

        for (int k = 0; k < lut->y_dim; k++) {
            lut->S[i][j + k] -= learning_rate * y_gradient[k];
        }
    }
}

/* build_Model: create every table and choose every random thing.
 *
 *   Token_embedder       random in [-1, 1], then frozen.
 *   FFN[l]               6 index bits -> 64 rows of 32 floats, per table.
 *   head V               6+6+4 = 16 index bits -> 65,536 rows of 32 floats,
 *                        per table.
 *   Positional_encoding  random in [-1, 1] (so the initial PE bits are
 *                        random); learnable.
 *   unembedder           6 index bits -> 64 rows of 256 floats (one logit
 *                        per byte).
 *
 * Compare these sizes with Table III ("Memory footprint" rows). There is no
 * srand(), so every run uses the same random numbers. */
void build_Model(Model *m) {

    random_vector(m->Token_embedder[0], VOCAB_SIZE * EMBEDDING_DIM, 1.0f);

    for (int l = 0; l < NUM_LAYERS; l++) {
        // the inputs are z_pos
        build_LUT(&m->FFN[l], N_C, EMBEDDING_DIM);
        for (int h = 0; h < NUM_HEADS; h++) {
            random_vector(m->head[l][h].Positional_encoding[0][0],
                          CONTEXT_SIZE * N_T * POSITIONAL_DIM, 1.0f);
            // the inputs are concatenated [Q, P, PE]
            build_LUT(&m->head[l][h].V, N_C + N_C + POSITIONAL_DIM,
                      EMBEDDING_DIM);
        }
    }
    build_LUT(&m->unembedder, N_C, VOCAB_SIZE);
}

void free_Model(Model *m) {

    for (int l = 0; l < NUM_LAYERS; l++) {
        free_LUT(&m->FFN[l]);
        for (int h = 0; h < NUM_HEADS; h++) {
            free_LUT(&m->head[l][h].V);
        }
    }
    free_LUT(&m->unembedder);
}

/* attention_forward: Eq. (15), using the V-index cache of
 * Sect. IV-D.
 *
 *   Step 1 (linear in context length): compute index bits ONCE per position
 *          (V_cache[pos]) and ONCE per distance (PE_cache[d]). In this loop
 *          PE_cache is indexed by `pos`, but it means distance d = pos:
 *          Positional_encoding[d] is "the PE for distance d".
 *   Step 2 (quadratic): for every pair pos1 < pos, glue the bits and add the
 *          row:
 *              y[pos] += sum_{pos1 < pos} V[z_pos, z_pos1, PE_{pos-pos1}]
 *
 * "Causal": a position only looks at EARLIER positions, so the prediction of
 * the next character can't cheat by seeing it. Position 0 has no earlier
 * positions and gets nothing from attention.
 *
 * Differences from a standard transformer:
 *   - No attention weights and no softmax: each pair directly contributes a
 *     whole vector, and the contributions are added. Sect. IV-C: "the
 *     attention network learns the relative magnitudes (attention scores) of
 *     value vectors".
 *   - No normalization: y[31] receives a sum of 31 pair contributions, y[1]
 *     only one.
 *
 * x is the input (a copy of z made by the caller); y is z itself (residual
 * "+="). */
void attention_forward(AttentionHead *head,
                       float x[CONTEXT_SIZE][EMBEDDING_DIM],
                       float y[CONTEXT_SIZE][EMBEDDING_DIM]) {

    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        cache_index(&head->V, &head->V_cache[pos], x[pos]);
        cache_PE_index(&head->PE_cache[pos], head->Positional_encoding[pos]);
    }

    for (int pos = 1; pos < CONTEXT_SIZE; pos++) {
        for (int pos1 = 0; pos1 < pos; pos1++) {
            concatenated_LUT_forward(&head->V, &head->V_cache[pos],
                                     &head->V_cache[pos1],
                                     &head->PE_cache[pos - pos1], y[pos]);
        }
    }
}

/* attention_backward: run the per-pair learning rule for every pair,
 * collecting
 *   - input gradients: into x_grad[pos] (as query) and x_grad[pos1] (as key);
 *   - positional-encoder gradients per distance: pos_grad[d];
 * and updating the V table rows along the way. Then the positional-encoder
 * floats take one gradient-descent step: PE -= learning_rate * dLoss/dPE.
 * The PE vectors are the only learned floats in the model that are not table
 * rows. */
void attention_backward(AttentionHead *head,
                        float x_grad[CONTEXT_SIZE][EMBEDDING_DIM],
                        float y_grad[CONTEXT_SIZE][EMBEDDING_DIM]) {

    float pos_grad[CONTEXT_SIZE][N_T][POSITIONAL_DIM];
    memset(pos_grad, 0, CONTEXT_SIZE * N_T * POSITIONAL_DIM * sizeof(float));

    for (int pos = 1; pos < CONTEXT_SIZE; pos++) {
        for (int pos1 = 0; pos1 < pos; pos1++) {
            concatenated_LUT_backward(
                &head->V, &head->V_cache[pos], &head->V_cache[pos1],
                &head->PE_cache[pos - pos1], x_grad[pos], x_grad[pos1],
                pos_grad[pos - pos1], y_grad[pos]);
        }
    }

    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        for (int i = 0; i < N_T; i++) {
            for (int k = 0; k < POSITIONAL_DIM; k++) {
                head->Positional_encoding[pos][i][k] -=
                    learning_rate * pos_grad[pos][i][k];
            }
        }
    }
}

/* model_forward: the whole prediction (Fig. 11b, bottom to top).
 * Before it is called, load_snippet() has put the token embeddings into m->z.
 *
 * Per layer:
 *   - All heads read the SAME snapshot x (a copy of z taken before any head
 *     runs) and all add into z. So heads run in parallel, not in a chain, and
 *     their outputs are simply added "without any weighting" (Sect. IV-C).
 *   - Then the FFN reads z[pos] and adds into z[pos]
 *     (Eq. 12: z_i = x_i + S_{x_i}). The paper notes the FFN is optional
 *     ("attention-only" model, Table III blue column).
 *
 * Finally the unembedder turns each z[pos] into 256 logits.
 *
 * The caches filled here (V_cache, PE_cache, FFN_cache, unembedder_vars) are
 * all the backward pass needs. The intermediate z vectors are overwritten and
 * not kept (Sect. IV-A: "we do not need to keep activation vectors"). */
void model_forward(Model *m) {

    for (int l = 0; l < NUM_LAYERS; l++) {

        // AttentionHead from all z to all z
        float x[CONTEXT_SIZE][EMBEDDING_DIM];
        // each head will be looking at the same input
        memcpy(x, m->z, CONTEXT_SIZE * EMBEDDING_DIM * sizeof(float));
        for (int h = 0; h < NUM_HEADS; h++) {
            // resnet connections: add to the output from the previous layer,
            // m->z
            attention_forward(&m->head[l][h], x, m->z);
        }

        // FFN from z_pos to z_pos.
        // For attention-only mode, comment the 4 lines below and 3 lines in
        // the next function
        for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
            // resnet connections: from m->z
            cache_index(&m->FFN[l], &m->FFN_cache[l][pos], m->z[pos]);
            // back to m->z
            LUT_forward(&m->FFN[l], &m->FFN_cache[l][pos], m->z[pos]);
        }
    }

    memset(m->output, 0, CONTEXT_SIZE * VOCAB_SIZE * sizeof(float));
    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        cache_index(&m->unembedder, &m->unembedder_vars[pos], m->z[pos]);
        LUT_forward(&m->unembedder, &m->unembedder_vars[pos], m->output[pos]);
    }
}

/* model_backward: backpropagation, top to bottom (Fig. 10 lines
 * 18-31). On entry m->output holds dLoss/dlogits (set up in
 * model_training_step).
 *
 * x_grad[pos] is the running dLoss/dz[pos]: "which way should z[pos] change
 * to lower the loss?". It starts at zero and is first filled by the
 * unembedder's backward pass.
 *
 * Residual connections and gradients, the key trick: if
 * z_out = z_in + f(z_in), then
 *
 *     dLoss/dz_in = dLoss/dz_out + (contribution through f)
 *
 * In code: y_grad = a copy of x_grad (= dLoss/dz_out), and x_grad is NOT
 * zeroed, which keeps the "dLoss/dz_out" term. The block's backward then ADDS
 * its contribution through f into x_grad. That's the meaning of "don't
 * zero-out x_grad, but add to it" (Fig. 10 line 22: v^l <- v^{l+1}).
 *
 * The order is the reverse of model_forward: unembedder, then for
 * l = 5..0: FFN, attention. Each block's backward uses the caches recorded at
 * exactly the z that block saw. */
void model_backward(Model *m) {

    float y_grad[CONTEXT_SIZE][EMBEDDING_DIM];
    float x_grad[CONTEXT_SIZE][EMBEDDING_DIM];
    memset(x_grad, 0, CONTEXT_SIZE * EMBEDDING_DIM * sizeof(float));

    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        LUT_backward(&m->unembedder, &m->unembedder_vars[pos], x_grad[pos],
                     m->output[pos]);
    }

    for (int l = NUM_LAYERS - 1; l >= 0; l--) {

        // FFN from z_pos to z_pos
        // don't zero-out x_grad, but add to it (resnet connections)
        memcpy(y_grad, x_grad, CONTEXT_SIZE * EMBEDDING_DIM * sizeof(float));
        // For attention-only mode, comment the 3 lines below and 4 lines in
        // the previous function
        for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
            LUT_backward(&m->FFN[l], &m->FFN_cache[l][pos], x_grad[pos],
                         y_grad[pos]);
        }

        // AttentionHead from all z to all z
        // don't zero-out x_grad, but add to it (resnet connections)
        memcpy(y_grad, x_grad, CONTEXT_SIZE * EMBEDDING_DIM * sizeof(float));
        for (int h = 0; h < NUM_HEADS; h++) {
            attention_backward(&m->head[l][h], x_grad, y_grad);
        }
    }

    // Here x_grad = dLoss/d(embedding of each input token). Applying
    // it would make the embedder learnable (Sect. IV-C says it is); it's
    // disabled (README deviation 4), so this gradient is computed and then
    // thrown away.
    // no need to compute gradients for the embedder; just update the synaptic
    // values
    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        for (int k = 0; k < EMBEDDING_DIM; k++) {
            // does not improve performance in character prediction mode, so
            // OK to disable
            // m->Token_embedder[m->tokens[pos]][k] -=
            //     learning_rate * x_grad[pos][k];
        }
    }
}

/* One learning step on the snippet currently loaded in m.
 *
 * The middle loop computes dLoss/dlogits for the cross-entropy loss, using a
 * standard identity: if p = softmax(logits) and Loss = -log p[correct], then
 *
 *     dLoss/dlogit_k = p_k - (1 if k == correct else 0).
 *
 * So "softmax, then subtract 1 at the correct token" turns m->output from
 * logits into exactly the gradient the unembedder's backward pass needs.
 * Intuitively: push the correct character's score up by (1 - p_correct), push
 * every other score k down by p_k.
 *
 * All 32 positions are trained at once: position pos predicts tokens[pos+1]
 * having seen tokens[0..pos]. The loss itself is never computed here, only
 * its gradient. */
void model_training_step(Model *m) {

    model_forward(m);
    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        softmax(m->output[pos], VOCAB_SIZE, 1.0f);
        m->output[pos][m->tokens[pos + 1]] -= 1.0f; // output become a gradient
    }
    model_backward(m);
}

/* ---------------- DATA HANDLING (no paper concepts here) ---------
 *
 *   data                  The entire training file in memory, as raw bytes.
 *   length                Number of valid snippet START positions
 *                         (file size - 33), so a 33-byte snippet starting
 *                         anywhere in [0, length) fits.
 *   reserved_for_testing  One flag per start position; 1 = "too close to a
 *                         test snippet, never train here" (so the model never
 *                         sees the test text).
 *   testing_input_data    The TESTING_LENGTH random start positions used for
 *                         validation.
 */
typedef struct {
    unsigned char *data;
    int length;
    unsigned char *reserved_for_testing;
    int testing_input_data[TESTING_LENGTH];

} TrainingData;

void free_training_data(TrainingData *training) {
    if (training->data) {
        free(training->data);
    }
    if (training->reserved_for_testing) {
        free(training->reserved_for_testing);
    }
}

void load_training_data(TrainingData *training, char *fname) {
    FILE *file = fopen(fname, "rb");
    if (!file) {
        printf("Error opening training datafile %s", fname);
        exit(1);
    } else {
        printf("Successfully opened training data file %s\n", fname);
    }

    // Get file size
    // `int` limits this to files under 2 GB.
    fseek(file, 0, SEEK_END);
    int file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the entire file
    unsigned char *local_data = (unsigned char *)malloc(file_size);
    if (!local_data) {
        perror("Error allocating memory");
        fclose(file);
        exit(1);
    }

    // Read the entire file
    training->length = fread(local_data, 1, file_size, file);
    fclose(file);

    if (training->length != file_size) {
        printf("Warning: Expected %d bytes but read %d bytes\n", file_size,
               training->length);
    }

    training->data = local_data;
    training->length -= CONTEXT_SIZE + 1;

    training->reserved_for_testing =
        (unsigned char *)calloc(training->length, sizeof(unsigned char));
    memset(training->reserved_for_testing, 0,
           training->length * sizeof(unsigned char));

    // Mark every start position within +-CONTEXT_SIZE of a test start
    // as reserved, so no training snippet overlaps a test snippet.
    // BUG: testing_input_data[i] + j can reach length-1+CONTEXT_SIZE, which
    // writes past the end of reserved_for_testing (allocated with `length`
    // bytes). The upper end needs clamping too, e.g.
    // MIN(training->length - 1, MAX(0, ...)).
    for (int i = 0; i < TESTING_LENGTH; i++) {
        training->testing_input_data[i] = rand() % training->length;
        // ensure that the testing input data is not too close to the training
        // data
        for (int j = -CONTEXT_SIZE; j <= CONTEXT_SIZE; j++) {
            training->reserved_for_testing[MAX(
                0, training->testing_input_data[i] + j)] = 1;
        }
    }
    printf("Successfully loaded training data\n");
}

// Pick a random snippet start that isn't reserved for testing (retry
// until found).
int get_random_training_index(TrainingData *training) {

    int symbol_index;
    do {
        symbol_index = rand() % training->length;
    } while (training->reserved_for_testing[symbol_index] == 1);
    return symbol_index;
}

// Embedding = plain table lookup: copy row text[pos] of
// Token_embedder into `input`.
void embed_token(Model *m, unsigned char *text, int pos, float *input) {

    memcpy(input, m->Token_embedder[text[pos]], EMBEDDING_DIM * sizeof(float));
}

/* Prepare the model's input for the snippet starting at char_start:
 * z[pos] = embedding of byte pos (the initial "spiking pattern" at each
 * position), and tokens[0..32] = the 33 raw bytes (tokens[32] is only ever a
 * target, never an input). */
void load_snippet(Model *m, TrainingData *training, int char_start) {

    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        embed_token(m, training->data + char_start, pos, m->z[pos]);
        m->tokens[pos] = (int)training->data[char_start + pos];
    }
    m->tokens[CONTEXT_SIZE] = (int)training->data[char_start + CONTEXT_SIZE];
}

/* Generate one character: run the model, take the probabilities at
 * the LAST position (the prediction for the character after the context),
 * sharpen them with temperature 0.4, and draw a random character from them.
 */
int model_inference(Model *m) {

    model_forward(m);
    softmax(m->output[CONTEXT_SIZE - 1], VOCAB_SIZE, 0.4f);
    int sampled_index = sample(m->output[CONTEXT_SIZE - 1], VOCAB_SIZE);
    return sampled_index;
}

/* Generate `response_length` characters after `prompt`
 * ("autoregressive" generation): predict one character, append it, drop the
 * oldest one, repeat -- a sliding 32-character window. No learning happens
 * here. strlcpy needs glibc >= 2.38 (or BSD/macOS). A prompt shorter than 32
 * characters would leave the rest of prompt_copy uninitialized (the author's
 * "Need to fix" note). */
void model_prompt_response(Model *m, unsigned char *prompt,
                           int response_length) {

    unsigned char prompt_copy[CONTEXT_SIZE + 1];
    // Need to fix: what if the prompt is too short?
    strlcpy((char *)prompt_copy, (const char *)prompt, CONTEXT_SIZE + 1);
    printf("%s", prompt_copy);

    for (int i = 0; i < response_length; i++) {
        for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
            embed_token(m, prompt_copy, pos, m->z[pos]);
        }
        int response = model_inference(m);
        printf("%c", (unsigned char)response);

        // shift the prompt by one character and insert response as the last
        // character
        for (int j = 0; j < CONTEXT_SIZE - 1; j++) {
            prompt_copy[j] = prompt_copy[j + 1];
        }
        prompt_copy[CONTEXT_SIZE - 1] = (unsigned char)response;
    }
}

// ============================================================================
// Main function
// ============================================================================
/* The training loop:
 *
 *   for 10^8 steps:
 *       pick a random training snippet; set the learning rate for step t;
 *       forward + backward + update  (model_training_step)
 *       every 10,000 steps:
 *           validation: average -log p(correct next char) at the LAST
 *           position over 10,000 held-out snippets (forward only, no
 *           learning) -> append to loss.csv;
 *           then print a sample of generated text.
 *
 * The validation loss is the number to watch. The paper's Fig. 13 plots
 * exp(loss) ("validation perplexity") against training examples (= t). */
int main(int argc, char *argv[]) {

    TrainingData training;
    // put your text file's name here
    load_training_data(&training, "train_v2_drcat_02.csv");

    // empties loss.csv
    FILE *file_loss = fopen(FILE_NAME, "w");
    fclose(file_loss);

    Model m;
    build_Model(&m);

    for (int t = 0; t < 100000000; t++) {

        load_snippet(&m, &training, get_random_training_index(&training));
        // Adam scheduler; uses `t`, see the macro's comment
        learning_rate = LEARNING_RATE;
        model_training_step(&m);

        if (t % 10000 == 0) {

            printf("...validating... ");
            fflush(stdout);
            float validation_loss = 0;

            for (int i = 0; i < TESTING_LENGTH; i++) {
                load_snippet(&m, &training, training.testing_input_data[i]);
                model_forward(&m);
                softmax(m.output[CONTEXT_SIZE - 1], VOCAB_SIZE, 1.0f);
                validation_loss +=
                    -log(m.output[CONTEXT_SIZE - 1][m.tokens[CONTEXT_SIZE]]);
            }
            // in nats; BPC = this / 0.693
            validation_loss /= TESTING_LENGTH;

            FILE *file_loss = fopen(FILE_NAME, "a");
            fprintf(file_loss, "%d, %f\n", t, validation_loss);
            fclose(file_loss);

            printf("\rt=%d,000, loss=%5.3f: ", t / 1000, validation_loss);
            model_prompt_response(
                &m, (unsigned char *)"insert your validation prompt here ", 80);
            printf("\n");
        }
        printf("\rt=%d", t);
        fflush(stdout);
    }

    free_Model(&m);
    free_training_data(&training);
    return 0;
}
