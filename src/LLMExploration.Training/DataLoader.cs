using LLMExploration.Core;

namespace LLMExploration.Training;

/// <summary>
/// Loads text data for training a language model.
///
/// The data loading pipeline:
///   1. Read raw text from file
///   2. Encode with character-level tokenizer to get an integer array
///   3. At each training step, randomly sample (batchSize) windows of (blockSize) tokens
///   4. Inputs are the windows; targets are the same windows shifted by 1
///
/// Example with blockSize=4 and text "hello":
///   Tokens: [h=0, e=1, l=2, l=2, o=3]
///   Possible windows (input -> target):
///     [0, 1, 2, 2] -> [1, 2, 2, 3]  (predict next char at each position)
///
/// This is "next token prediction" - the simplest and most powerful self-supervised
/// learning objective for language modeling.
/// </summary>
public class DataLoader
{
    private readonly int[] _tokens;
    private readonly int _blockSize;
    private readonly Random _rng;

    /// <summary>Total number of tokens in the dataset.</summary>
    public int TokenCount => _tokens.Length;

    /// <summary>Maximum valid starting position for a window.</summary>
    private int MaxStart => _tokens.Length - _blockSize - 1;

    public DataLoader(string filePath, Tokenizer tokenizer, int blockSize, int? seed = null)
    {
        _blockSize = blockSize;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();

        var text = File.ReadAllText(filePath);
        _tokens = tokenizer.Encode(text);

        if (_tokens.Length < blockSize + 1)
            throw new ArgumentException(
                $"Text has only {_tokens.Length} tokens, need at least {blockSize + 1} for blockSize={blockSize}");

        Console.WriteLine($"[DataLoader] Loaded {_tokens.Length:N0} tokens from '{filePath}'");
    }

    /// <summary>
    /// Create a DataLoader from an already-encoded token array.
    /// </summary>
    public DataLoader(int[] tokens, int blockSize, int? seed = null)
    {
        _tokens = tokens;
        _blockSize = blockSize;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Sample a random batch of (input, target) pairs.
    ///
    /// Each sample is a window of (blockSize) consecutive tokens.
    /// The target is the same window shifted right by 1:
    ///   input[t] = tokens[start + t]
    ///   target[t] = tokens[start + t + 1]
    ///
    /// This means at each position t, the model should predict target[t]
    /// given input[0..t] (causal prediction).
    ///
    /// Returns:
    ///   inputs:  int[batchSize][blockSize]
    ///   targets: int[batchSize][blockSize]
    /// </summary>
    public (int[][] inputs, int[][] targets) GetBatch(int batchSize)
    {
        int maxStart = MaxStart;
        if (maxStart <= 0)
            throw new InvalidOperationException("Not enough data for the given blockSize");

        var inputs  = new int[batchSize][];
        var targets = new int[batchSize][];

        for (int b = 0; b < batchSize; b++)
        {
            // Random starting position
            int start = _rng.Next(0, maxStart + 1);

            inputs[b]  = new int[_blockSize];
            targets[b] = new int[_blockSize];

            for (int t = 0; t < _blockSize; t++)
            {
                inputs[b][t]  = _tokens[start + t];
                targets[b][t] = _tokens[start + t + 1];
            }
        }

        return (inputs, targets);
    }

    /// <summary>
    /// Get the token array for examining data distribution.
    /// </summary>
    public int[] GetTokens() => _tokens;
}
