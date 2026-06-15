using System.Text.Json;

namespace LLMExploration.Core;

/// <summary>
/// Configuration for the GPT model architecture.
///
/// These hyperparameters control the capacity and behavior of the model.
/// The defaults are intentionally small so training runs in minutes on CPU.
///
/// Key relationships:
///   - HeadDim = NEmbedding / NHead  (must divide evenly)
///   - MLP hidden size = 4 * NEmbedding  (standard GPT ratio)
///   - Total parameters ≈ VocabSize*NEmbedding + NLayer*(12*NEmbedding^2) + NEmbedding*VocabSize
/// </summary>
public class GPTConfig
{
    /// <summary>
    /// Number of unique tokens in the vocabulary.
    /// For character-level tokenization this equals the number of unique characters.
    /// GPT-2 small uses 50257 (BPE subwords).
    /// </summary>
    public int VocabSize { get; set; } = 65;

    /// <summary>
    /// Maximum sequence length (context window) in tokens.
    /// The causal mask ensures token i can only attend to tokens 0..i.
    /// GPT-2 uses 1024; we use 64 for fast CPU training.
    /// </summary>
    public int BlockSize { get; set; } = 64;

    /// <summary>
    /// Embedding dimension: the width of every hidden representation.
    /// All layers maintain this dimensionality (residual stream).
    /// GPT-2 small uses 768; we use 64.
    /// </summary>
    public int NEmbedding { get; set; } = 64;

    /// <summary>
    /// Number of attention heads. Each head independently attends over the sequence
    /// with dimension NEmbedding/NHead. Using multiple heads lets the model
    /// simultaneously attend to different positions for different reasons.
    /// HeadDim = NEmbedding / NHead must be an integer.
    /// </summary>
    public int NHead { get; set; } = 4;

    /// <summary>
    /// Number of transformer blocks stacked in sequence.
    /// Each block alternates between attention (communication) and MLP (computation).
    /// GPT-2 small uses 12; we use 4.
    /// </summary>
    public int NLayer { get; set; } = 4;

    /// <summary>Convenience property: embedding dimension per attention head.</summary>
    public int HeadDim => NEmbedding / NHead;

    /// <summary>Save config to JSON file.</summary>
    public void Save(string path)
    {
        var json = JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    /// <summary>Load config from JSON file.</summary>
    public static GPTConfig Load(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<GPTConfig>(json)
            ?? throw new InvalidDataException("Failed to parse GPTConfig JSON");
    }
}
