namespace LLMExploration.Core.Layers;

/// <summary>
/// Multi-Head Causal (Masked) Self-Attention.
///
/// This is the core mechanism that allows tokens to communicate with each other.
/// "Causal" means token i can only attend to tokens 0..i (no future leakage).
/// "Multi-head" means we run H parallel attention computations, each focusing
/// on different aspects of the relationships between tokens.
///
/// Algorithm:
///   1. Project input x[B,T,C] into Q, K, V each [B,T,C] via learned linear layers
///   2. Reshape to [B,H,T,d] where d = C/H (head dimension)
///   3. Compute attention scores: scores[b,h,i,j] = Q[b,h,i,:] . K[b,h,j,:] / sqrt(d)
///      - The scale 1/sqrt(d) prevents dot products from growing too large
///   4. Apply causal mask: set scores[b,h,i,j] = -inf for j > i
///      - After softmax, these become 0, so token i ignores future tokens j
///   5. Softmax over j: attn[b,h,i,:] = softmax(scores[b,h,i,:])
///   6. Weighted sum of values: out[b,h,i,:] = sum_j attn[b,h,i,j] * V[b,h,j,:]
///   7. Reshape back to [B,T,C] and project through output linear
///
/// Intuition: Q="what am I looking for", K="what do I contain", V="what I'll share"
/// </summary>
public class CausalSelfAttention : Module
{
    private readonly int _nHead;
    private readonly int _headDim;
    private readonly int _nEmbd;
    private readonly float _scale;

    // Separate Q, K, V projections (no bias, following GPT-2)
    public Linear QueryProj { get; }
    public Linear KeyProj { get; }
    public Linear ValueProj { get; }
    public Linear OutProj { get; }

    // Cache for backward pass
    private Tensor? _lastInput;
    private Tensor? _Q, _K, _V;           // [B, H, T, d] after reshape
    private Tensor? _attnWeights;          // [B, H, T, T] after softmax
    private int _lastB, _lastT;

    public CausalSelfAttention(int nEmbd, int nHead, System.Random? rng = null)
    {
        _nEmbd = nEmbd;
        _nHead = nHead;
        _headDim = nEmbd / nHead;
        _scale = 1f / MathF.Sqrt(_headDim);

        // No bias in attention projections (common in GPT architectures)
        QueryProj = new Linear(nEmbd, nEmbd, useBias: false, rng: rng);
        KeyProj   = new Linear(nEmbd, nEmbd, useBias: false, rng: rng);
        ValueProj = new Linear(nEmbd, nEmbd, useBias: false, rng: rng);
        OutProj   = new Linear(nEmbd, nEmbd, useBias: true,  rng: rng);
    }

    /// <summary>
    /// Forward pass: compute multi-head causal self-attention.
    /// Input x: [B, T, C] where B=batch, T=sequence length, C=embedding dim
    /// Returns: [B, T, C]
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        _lastInput = x;
        int B = x.Shape[0], T = x.Shape[1], C = _nEmbd;
        int H = _nHead, d = _headDim;
        _lastB = B; _lastT = T;

        // Step 1: Project to Q, K, V  [B, T, C] each
        var qFlat = QueryProj.Forward(x);  // [B, T, C]
        var kFlat = KeyProj.Forward(x);
        var vFlat = ValueProj.Forward(x);

        // Step 2: Reshape to [B, H, T, d] by rearranging dims
        // qFlat[b, t, h*d + i] -> Q[b, h, t, i]
        _Q = ReshapeToHeads(qFlat, B, T, H, d);
        _K = ReshapeToHeads(kFlat, B, T, H, d);
        _V = ReshapeToHeads(vFlat, B, T, H, d);

        // Step 3: Attention scores = Q @ K^T * scale  => [B, H, T, T]
        var scores = new Tensor(new[] { B, H, T, T });
        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
        {
            float dot = 0f;
            for (int k = 0; k < d; k++)
                dot += _Q[b, h, i, k] * _K[b, h, j, k];
            scores[b, h, i, j] = dot * _scale;
        }

        // Step 4: Causal mask - token i cannot attend to future tokens j > i
        // Set masked positions to -inf so softmax gives 0
        const float NEG_INF = -1e9f;
        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        for (int i = 0; i < T; i++)
        for (int j = i + 1; j < T; j++)  // j > i means future
            scores[b, h, i, j] = NEG_INF;

        // Step 5: Softmax over j dimension => attention weights [B, H, T, T]
        _attnWeights = new Tensor(new[] { B, H, T, T });
        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        for (int i = 0; i < T; i++)
        {
            // Numerically stable softmax: subtract max before exp
            float maxVal = float.NegativeInfinity;
            for (int j = 0; j <= i; j++)
                maxVal = MathF.Max(maxVal, scores[b, h, i, j]);

            float sumExp = 0f;
            for (int j = 0; j <= i; j++)
            {
                float e = MathF.Exp(scores[b, h, i, j] - maxVal);
                _attnWeights[b, h, i, j] = e;
                sumExp += e;
            }
            // Normalize and set future positions to 0
            for (int j = 0; j < T; j++)
            {
                if (j <= i)
                    _attnWeights[b, h, i, j] /= sumExp;
                else
                    _attnWeights[b, h, i, j] = 0f;
            }
        }

        // Step 6: Weighted sum of values: attnOut[b,h,i,k] = sum_j attn[b,h,i,j] * V[b,h,j,k]
        var attnOut = new Tensor(new[] { B, H, T, d });
        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        for (int i = 0; i < T; i++)
        for (int k = 0; k < d; k++)
        {
            float sum = 0f;
            for (int j = 0; j < T; j++)
                sum += _attnWeights[b, h, i, j] * _V[b, h, j, k];
            attnOut[b, h, i, k] = sum;
        }

        // Step 7: Reshape back to [B, T, C] and project
        var attnOutFlat = ReshapeFromHeads(attnOut, B, T, H, d);
        return OutProj.Forward(attnOutFlat);
    }

    /// <summary>
    /// Backward pass through multi-head causal self-attention.
    ///
    /// The backward pass reverses each step of the forward pass:
    ///   1. Backward through output projection -> dAttnOutFlat
    ///   2. Reshape dAttnOutFlat to [B,H,T,d]
    ///   3. Backward through weighted sum:
    ///      dAttnWeights[b,h,i,j] = sum_k dAttnOut[b,h,i,k] * V[b,h,j,k]
    ///      dV[b,h,j,k]           = sum_i dAttnOut[b,h,i,k] * attnWeights[b,h,i,j]
    ///   4. Backward through softmax (with causal masking)
    ///   5. Backward through QK^T matmul:
    ///      dQ[b,h,i,k] = sum_j dScores[b,h,i,j] * K[b,h,j,k] * scale
    ///      dK[b,h,j,k] = sum_i dScores[b,h,i,j] * Q[b,h,i,k] * scale
    ///   6. Reshape and backward through Q, K, V projections
    /// </summary>
    public Tensor Backward(Tensor dOut)
    {
        if (_lastInput == null || _Q == null || _K == null || _V == null || _attnWeights == null)
            throw new InvalidOperationException("Must call Forward before Backward");

        int B = _lastB, T = _lastT, C = _nEmbd, H = _nHead, d = _headDim;

        // Step 1: Backward through output projection
        var dAttnOutFlat = OutProj.Backward(dOut);  // [B, T, C]

        // Step 2: Reshape to [B, H, T, d]
        var dAttnOut = ReshapeToHeads(dAttnOutFlat, B, T, H, d);

        // Step 3: Backward through V-weighted sum
        // dAttnWeights[b,h,i,j] = sum_k dAttnOut[b,h,i,k] * V[b,h,j,k]
        // dV[b,h,j,k]           = sum_i attnWeights[b,h,i,j] * dAttnOut[b,h,i,k]
        var dAttnWeights = new Tensor(new[] { B, H, T, T });
        var dV = new Tensor(new[] { B, H, T, d });

        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {
            for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++)
            {
                float sum = 0f;
                for (int k = 0; k < d; k++)
                    sum += dAttnOut[b, h, i, k] * _V[b, h, j, k];
                dAttnWeights[b, h, i, j] = sum;
            }

            for (int j = 0; j < T; j++)
            for (int k = 0; k < d; k++)
            {
                float sum = 0f;
                for (int i = 0; i < T; i++)
                    sum += _attnWeights[b, h, i, j] * dAttnOut[b, h, i, k];
                dV[b, h, j, k] = sum;
            }
        }

        // Step 4: Backward through softmax
        // For each row i: dScores[i,j] = attn[i,j] * (dAttnWeights[i,j] - dot(dAttnWeights[i,:], attn[i,:]))
        var dScores = new Tensor(new[] { B, H, T, T });
        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        for (int i = 0; i < T; i++)
        {
            // Compute dot(dAttnWeights[i,:], attn[i,:]) - only up to i (causal)
            float dot = 0f;
            for (int j = 0; j <= i; j++)
                dot += dAttnWeights[b, h, i, j] * _attnWeights[b, h, i, j];

            for (int j = 0; j <= i; j++)
            {
                float attn_ij = _attnWeights[b, h, i, j];
                dScores[b, h, i, j] = attn_ij * (dAttnWeights[b, h, i, j] - dot);
            }
            // j > i: already 0, future positions are masked
        }

        // Step 5: Backward through QK^T * scale
        // scores[b,h,i,j] = (sum_k Q[b,h,i,k] * K[b,h,j,k]) * scale
        // dQ[b,h,i,k] = sum_j dScores[b,h,i,j] * K[b,h,j,k] * scale
        // dK[b,h,j,k] = sum_i dScores[b,h,i,j] * Q[b,h,i,k] * scale
        var dQ = new Tensor(new[] { B, H, T, d });
        var dK = new Tensor(new[] { B, H, T, d });

        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {
            for (int i = 0; i < T; i++)
            for (int k = 0; k < d; k++)
            {
                float sum = 0f;
                for (int j = 0; j <= i; j++)  // causal: only past positions contributed
                    sum += dScores[b, h, i, j] * _K[b, h, j, k];
                dQ[b, h, i, k] = sum * _scale;
            }

            for (int j = 0; j < T; j++)
            for (int k = 0; k < d; k++)
            {
                float sum = 0f;
                for (int i = j; i < T; i++)  // token i attended to token j when i >= j
                    sum += dScores[b, h, i, j] * _Q[b, h, i, k];
                dK[b, h, j, k] = sum * _scale;
            }
        }

        // Step 6: Reshape gradients back to [B, T, C]
        var dQFlat = ReshapeFromHeads(dQ, B, T, H, d);
        var dKFlat = ReshapeFromHeads(dK, B, T, H, d);
        var dVFlat = ReshapeFromHeads(dV, B, T, H, d);

        // Step 7: Backward through Q, K, V projections
        var dXq = QueryProj.Backward(dQFlat);
        var dXk = KeyProj.Backward(dKFlat);
        var dXv = ValueProj.Backward(dVFlat);

        // Sum contributions to input gradient
        var dInput = new Tensor(_lastInput.Shape);
        for (int i = 0; i < dInput.Size; i++)
            dInput.Data[i] = dXq.Data[i] + dXk.Data[i] + dXv.Data[i];

        return dInput;
    }

    /// <summary>
    /// Reshape [B, T, H*d] to [B, H, T, d] by transposing the H and T dimensions.
    /// This groups each head's Q/K/V vectors together.
    ///
    /// x[b, t, h*d + i] -> out[b, h, t, i]
    /// </summary>
    private static Tensor ReshapeToHeads(Tensor x, int B, int T, int H, int d)
    {
        var result = new Tensor(new[] { B, H, T, d });
        for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
        for (int i = 0; i < d; i++)
        {
            result[b, h, t, i] = x.Data[b * T * H * d + t * H * d + h * d + i];
        }
        return result;
    }

    /// <summary>
    /// Reshape [B, H, T, d] back to [B, T, H*d].
    /// Reverses ReshapeToHeads.
    ///
    /// x[b, h, t, i] -> out[b, t, h*d + i]
    /// </summary>
    private static Tensor ReshapeFromHeads(Tensor x, int B, int T, int H, int d)
    {
        var result = new Tensor(new[] { B, T, H * d });
        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
        {
            result.Data[b * T * H * d + t * H * d + h * d + i] = x[b, h, t, i];
        }
        return result;
    }

    public override IEnumerable<Tensor> Parameters()
    {
        foreach (var p in QueryProj.Parameters()) yield return p;
        foreach (var p in KeyProj.Parameters()) yield return p;
        foreach (var p in ValueProj.Parameters()) yield return p;
        foreach (var p in OutProj.Parameters()) yield return p;
    }
}
