namespace LLMExploration.Core.Layers;

/// <summary>
/// Token + Positional Embedding layer.
///
/// Converts a sequence of integer token IDs into continuous vectors by:
/// 1. Looking up each token in a learned token embedding table
/// 2. Adding a learned positional embedding for each position
///
/// This gives the model two pieces of information:
///   - WHAT each token is (token embedding)
///   - WHERE each token is in the sequence (positional embedding)
///
/// Unlike sine/cosine positional encodings (original "Attention is All You Need"),
/// GPT uses learned positional embeddings - both tables are optimized by gradient descent.
///
/// Output shape: [T, NEmbedding] where T = sequence length
/// </summary>
public class Embedding : Module
{
    /// <summary>Token embedding table. Shape: [VocabSize, NEmbedding]</summary>
    public Tensor TokenWeight { get; }

    /// <summary>Position embedding table. Shape: [BlockSize, NEmbedding]</summary>
    public Tensor PosWeight { get; }

    private readonly int _vocabSize;
    private readonly int _blockSize;
    private readonly int _nEmbd;

    // Cache for backward pass
    private int[]? _lastTokenIds;

    public Embedding(int vocabSize, int blockSize, int nEmbd, System.Random? rng = null)
    {
        _vocabSize = vocabSize;
        _blockSize = blockSize;
        _nEmbd = nEmbd;

        // Initialize with small random values - standard for embeddings
        // std=0.02 is the GPT-2 initialization scheme
        TokenWeight = Tensor.Random(new[] { vocabSize, nEmbd }, std: 0.02f, requiresGrad: true, rng: rng);
        PosWeight = Tensor.Random(new[] { blockSize, nEmbd }, std: 0.02f, requiresGrad: true, rng: rng);
    }

    /// <summary>
    /// Forward pass: embed tokens and add positional embeddings.
    ///
    /// For each position t and token tokenIds[t]:
    ///   output[t, :] = TokenWeight[tokenIds[t], :] + PosWeight[t, :]
    ///
    /// This is a simple table lookup - no matrix multiplication needed.
    /// The gradient flows back through this lookup by accumulating into
    /// the relevant rows of the embedding tables.
    /// </summary>
    /// <param name="tokenIds">Integer token ids, shape [T]</param>
    /// <returns>Embedded vectors, shape [T, NEmbedding]</returns>
    public Tensor Forward(int[] tokenIds)
    {
        int T = tokenIds.Length;
        if (T > _blockSize)
            throw new ArgumentException($"Sequence length {T} exceeds blockSize {_blockSize}");

        _lastTokenIds = tokenIds;

        var output = new Tensor(new[] { T, _nEmbd });

        for (int t = 0; t < T; t++)
        {
            int tok = tokenIds[t];
            for (int c = 0; c < _nEmbd; c++)
            {
                // Token embedding lookup + positional embedding lookup
                output.Data[t * _nEmbd + c] =
                    TokenWeight.Data[tok * _nEmbd + c] +
                    PosWeight.Data[t * _nEmbd + c];
            }
        }

        return output;
    }

    /// <summary>
    /// Backward pass: scatter gradients back into the embedding tables.
    ///
    /// Since embedding is just a lookup, the gradient for token t flows to:
    ///   TokenWeight.Grad[tokenIds[t], :] += dOut[t, :]
    ///   PosWeight.Grad[t, :]             += dOut[t, :]
    ///
    /// Note: multiple positions may have the same token id, so we accumulate (+=).
    /// </summary>
    /// <param name="dOut">Gradient flowing back, shape [T, NEmbedding]</param>
    public void Backward(Tensor dOut)
    {
        if (_lastTokenIds == null)
            throw new InvalidOperationException("Must call Forward before Backward");

        int T = _lastTokenIds.Length;

        for (int t = 0; t < T; t++)
        {
            int tok = _lastTokenIds[t];
            for (int c = 0; c < _nEmbd; c++)
            {
                float grad = dOut.Data[t * _nEmbd + c];
                // Accumulate into token embedding gradient
                TokenWeight.Grad![tok * _nEmbd + c] += grad;
                // Accumulate into positional embedding gradient
                PosWeight.Grad![t * _nEmbd + c] += grad;
            }
        }
    }

    public override IEnumerable<Tensor> Parameters()
    {
        yield return TokenWeight;
        yield return PosWeight;
    }
}
