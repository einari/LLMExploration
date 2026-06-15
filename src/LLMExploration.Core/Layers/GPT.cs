using System.Text;
using System.Text.Json;

namespace LLMExploration.Core.Layers;

/// <summary>
/// The full GPT (Generative Pre-trained Transformer) model.
///
/// Architecture overview:
///   tokens -> Embedding (token + positional) -> [Block x NLayer] -> LayerNorm -> Linear (lm_head) -> logits
///
/// The model is trained to predict the next token at every position.
/// Given tokens [t0, t1, t2, ...], it outputs logits for what comes after each token.
/// The cross-entropy loss is averaged over all positions.
///
/// At inference time, we autoregressively generate:
///   1. Feed current context
///   2. Take logits for the last position
///   3. Sample next token
///   4. Append to context and repeat
///
/// This implementation processes one sequence at a time (B=1) for simplicity.
/// A production implementation would batch multiple sequences.
///
/// Parameter count (approximate):
///   Embedding: VocabSize*C + BlockSize*C
///   Per block: C^2 * 4 (attention) + C^2 * 8 (mlp) + 4*C (layernorms)
///   Final norm: 2*C
///   LM head: VocabSize*C (tied with token embedding in some implementations)
///   Total: ~NLayer * 12 * C^2 for large C
/// </summary>
public class GPT : Module
{
    private readonly GPTConfig _config;
    private readonly Embedding _embedding;
    private readonly Block[] _blocks;
    private readonly LayerNorm _lnF;
    private readonly Linear _lmHead;

    // Cache for backward pass
    private Tensor? _embOut;
    private Tensor[]? _blockOuts;
    private Tensor? _lnFOut;
    private Tensor? _logits;

    public GPT(GPTConfig config, System.Random? rng = null)
    {
        _config = config;
        _embedding = new Embedding(config.VocabSize, config.BlockSize, config.NEmbedding, rng);
        _blocks = new Block[config.NLayer];
        for (int i = 0; i < config.NLayer; i++)
            _blocks[i] = new Block(config.NEmbedding, config.NHead, rng);
        _lnF = new LayerNorm(config.NEmbedding);
        _lmHead = new Linear(config.NEmbedding, config.VocabSize, useBias: false, rng: rng);
    }

    /// <summary>
    /// Forward pass: token ids -> logits over vocabulary.
    ///
    /// Input: tokens array of length T (integer token ids)
    /// Output: logits tensor [1, T, VocabSize]
    ///
    /// The logits represent unnormalized log-probabilities for the next token
    /// at each position. To get probabilities: softmax(logits, dim=-1).
    /// </summary>
    public Tensor Forward(int[] tokens)
    {
        int T = tokens.Length;

        // Embed tokens: [T, C] -> unsqueeze to [1, T, C]
        var embFlat = _embedding.Forward(tokens);  // [T, C]

        // Add batch dimension: [T, C] -> [1, T, C]
        _embOut = new Tensor(new[] { 1, T, _config.NEmbedding });
        Array.Copy(embFlat.Data, _embOut.Data, embFlat.Size);

        // Pass through transformer blocks
        _blockOuts = new Tensor[_config.NLayer];
        Tensor x = _embOut;
        for (int i = 0; i < _config.NLayer; i++)
        {
            x = _blocks[i].Forward(x);
            _blockOuts[i] = x;
        }

        // Final layer norm: [1, T, C]
        _lnFOut = _lnF.Forward(x);

        // Project to vocabulary: [1, T, C] -> [1, T, VocabSize]
        _logits = _lmHead.Forward(_lnFOut);

        return _logits;
    }

    /// <summary>
    /// Compute forward pass and cross-entropy loss.
    ///
    /// The cross-entropy loss is:
    ///   loss = -(1/T) * sum_t log(softmax(logits[0,t,:])[targets[t]])
    ///
    /// This measures how surprised the model is by the actual next tokens.
    /// A perfect model would assign probability 1.0 to the correct token,
    /// giving loss = 0. A random model with vocab size V gives loss = log(V).
    /// </summary>
    /// <param name="tokens">Input token sequence [T]</param>
    /// <param name="targets">Target token sequence [T] (next tokens, shifted by 1)</param>
    /// <returns>logits [1,T,VocabSize] and mean cross-entropy loss</returns>
    public (Tensor logits, float loss) Loss(int[] tokens, int[] targets)
    {
        var logits = Forward(tokens);
        int T = tokens.Length;
        int V = _config.VocabSize;

        float totalLoss = 0f;
        for (int t = 0; t < T; t++)
        {
            // Find max for numerical stability
            float maxLogit = float.NegativeInfinity;
            for (int v = 0; v < V; v++)
                maxLogit = MathF.Max(maxLogit, logits[0, t, v]);

            // Softmax denominator
            float sumExp = 0f;
            for (int v = 0; v < V; v++)
                sumExp += MathF.Exp(logits[0, t, v] - maxLogit);

            // Log-probability of the correct token
            float logProb = logits[0, t, targets[t]] - maxLogit - MathF.Log(sumExp);
            totalLoss -= logProb;
        }

        return (logits, totalLoss / T);
    }

    /// <summary>
    /// Backward pass: compute gradients starting from the cross-entropy loss.
    ///
    /// The gradient of cross-entropy loss w.r.t. logits (before softmax) is:
    ///   dLogits[t, v] = (softmax(logits[t,:])[v] - onehot(target[t])[v]) / T
    ///
    /// This is a beautifully simple formula: increase probability of the correct token,
    /// decrease probability of all others, in proportion to their current probability.
    ///
    /// After computing dLogits, we backprop through:
    ///   lmHead -> lnF -> blocks (in reverse) -> embedding
    /// </summary>
    public void Backward(int[] targets)
    {
        if (_logits == null || _lnFOut == null || _blockOuts == null || _embOut == null)
            throw new InvalidOperationException("Must call Forward (via Loss) before Backward");

        int T = targets.Length;
        int V = _config.VocabSize;

        // Compute gradient of loss w.r.t. logits
        var dLogits = new Tensor(_logits.Shape);
        for (int t = 0; t < T; t++)
        {
            // Numerically stable softmax
            float maxLogit = float.NegativeInfinity;
            for (int v = 0; v < V; v++)
                maxLogit = MathF.Max(maxLogit, _logits[0, t, v]);

            float sumExp = 0f;
            var probs = new float[V];
            for (int v = 0; v < V; v++)
            {
                probs[v] = MathF.Exp(_logits[0, t, v] - maxLogit);
                sumExp += probs[v];
            }

            for (int v = 0; v < V; v++)
            {
                // dL/dlogit = (prob - onehot) / T
                float softmax = probs[v] / sumExp;
                dLogits[0, t, v] = (softmax - (v == targets[t] ? 1f : 0f)) / T;
            }
        }

        // Backward through lm_head
        var dLnFOut = _lmHead.Backward(dLogits);

        // Backward through final layer norm
        var dBlockOut = _lnF.Backward(dLnFOut);

        // Backward through transformer blocks in reverse order
        for (int i = _config.NLayer - 1; i >= 0; i--)
            dBlockOut = _blocks[i].Backward(dBlockOut);

        // Backward through embedding
        // Remove batch dim: [1, T, C] -> [T, C]
        var dEmbFlat = new Tensor(new[] { T, _config.NEmbedding });
        Array.Copy(dBlockOut.Data, dEmbFlat.Data, dEmbFlat.Size);
        _embedding.Backward(dEmbFlat);
    }

    /// <summary>
    /// Generate new text autoregressively.
    ///
    /// Starting from a context, repeatedly:
    ///   1. Run forward pass on current context (clipped to blockSize)
    ///   2. Extract logits for the last token position
    ///   3. Apply temperature scaling (higher = more random)
    ///   4. Sample from the softmax distribution
    ///   5. Append the sampled token and continue
    ///
    /// Temperature controls randomness:
    ///   - temperature=1.0: sample from the model's distribution
    ///   - temperature<1.0: sharpen distribution (more predictable)
    ///   - temperature>1.0: flatten distribution (more creative/random)
    /// </summary>
    public string Generate(int[] context, int maxNew, float temperature, Tokenizer tokenizer, System.Random? rng = null)
    {
        rng ??= new System.Random();
        var tokens = new List<int>(context);
        var generated = new List<int>();

        for (int step = 0; step < maxNew; step++)
        {
            // Clip context to blockSize (sliding window)
            int start = Math.Max(0, tokens.Count - _config.BlockSize);
            var ctx = tokens.GetRange(start, tokens.Count - start).ToArray();

            // Forward pass
            var logits = Forward(ctx);
            int lastT = ctx.Length - 1;
            int V = _config.VocabSize;

            // Extract logits for the last position and apply temperature
            var lastLogits = new float[V];
            for (int v = 0; v < V; v++)
                lastLogits[v] = logits[0, lastT, v] / temperature;

            // Numerically stable softmax to get probabilities
            float maxLogit = lastLogits.Max();
            float sumExp = 0f;
            var probs = new float[V];
            for (int v = 0; v < V; v++)
            {
                probs[v] = MathF.Exp(lastLogits[v] - maxLogit);
                sumExp += probs[v];
            }
            for (int v = 0; v < V; v++)
                probs[v] /= sumExp;

            // Sample from the distribution (multinomial sampling)
            float u = (float)rng.NextDouble();
            float cumulative = 0f;
            int nextToken = V - 1;
            for (int v = 0; v < V; v++)
            {
                cumulative += probs[v];
                if (u < cumulative)
                {
                    nextToken = v;
                    break;
                }
            }

            tokens.Add(nextToken);
            generated.Add(nextToken);
        }

        return tokenizer.Decode(generated.ToArray());
    }

    /// <summary>
    /// Save model to a binary file.
    ///
    /// Format:
    ///   - Magic number (4 bytes): 0x4C4C4D20 ("LLM ")
    ///   - Config JSON (4-byte length prefix + UTF-8 bytes)
    ///   - For each parameter: name (4-byte length + bytes), rank (4 bytes),
    ///     shape (rank * 4 bytes), data (size * 4 bytes)
    /// </summary>
    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        using var writer = new BinaryWriter(File.Open(path, FileMode.Create));

        // Magic number
        writer.Write(0x4C4C4D20);

        // Config as JSON
        var configJson = JsonSerializer.Serialize(_config);
        var configBytes = Encoding.UTF8.GetBytes(configJson);
        writer.Write(configBytes.Length);
        writer.Write(configBytes);

        // Write all parameters with their names
        var namedParams = GetNamedParameters();
        foreach (var (name, tensor) in namedParams)
        {
            var nameBytes = Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
            writer.Write(tensor.Shape.Length);
            foreach (int s in tensor.Shape)
                writer.Write(s);
            foreach (float f in tensor.Data)
                writer.Write(f);
        }
    }

    /// <summary>Load model weights from a binary file.</summary>
    public static GPT Load(string path)
    {
        using var reader = new BinaryReader(File.Open(path, FileMode.Open));

        int magic = reader.ReadInt32();
        if (magic != 0x4C4C4D20)
            throw new InvalidDataException("Invalid model file magic number");

        int configLen = reader.ReadInt32();
        var configBytes = reader.ReadBytes(configLen);
        var config = JsonSerializer.Deserialize<GPTConfig>(Encoding.UTF8.GetString(configBytes))
            ?? throw new InvalidDataException("Failed to parse config");

        var model = new GPT(config);
        var namedParams = model.GetNamedParameters().ToDictionary(x => x.name, x => x.tensor);

        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            int nameLen = reader.ReadInt32();
            var name = Encoding.UTF8.GetString(reader.ReadBytes(nameLen));
            int rank = reader.ReadInt32();
            int size = 1;
            for (int i = 0; i < rank; i++) size *= reader.ReadInt32();

            if (namedParams.TryGetValue(name, out var tensor))
            {
                for (int i = 0; i < size; i++)
                    tensor.Data[i] = reader.ReadSingle();
            }
            else
            {
                // Skip unknown parameter
                reader.ReadBytes(size * 4);
            }
        }

        return model;
    }

    /// <summary>
    /// Get all parameters with descriptive names for save/load.
    /// </summary>
    private IEnumerable<(string name, Tensor tensor)> GetNamedParameters()
    {
        yield return ("embedding.token", _embedding.TokenWeight);
        yield return ("embedding.pos", _embedding.PosWeight);
        for (int i = 0; i < _config.NLayer; i++)
        {
            var b = _blocks[i];
            yield return ($"block{i}.ln1.gamma", b.Ln1.Gamma);
            yield return ($"block{i}.ln1.beta", b.Ln1.Beta);
            yield return ($"block{i}.attn.q.weight", b.Attn.QueryProj.Weight);
            yield return ($"block{i}.attn.k.weight", b.Attn.KeyProj.Weight);
            yield return ($"block{i}.attn.v.weight", b.Attn.ValueProj.Weight);
            yield return ($"block{i}.attn.out.weight", b.Attn.OutProj.Weight);
            if (b.Attn.OutProj.Bias != null)
                yield return ($"block{i}.attn.out.bias", b.Attn.OutProj.Bias);
            yield return ($"block{i}.ln2.gamma", b.Ln2.Gamma);
            yield return ($"block{i}.ln2.beta", b.Ln2.Beta);
            yield return ($"block{i}.mlp.fc1.weight", b.Mlp.Fc1.Weight);
            yield return ($"block{i}.mlp.fc1.bias", b.Mlp.Fc1.Bias!);
            yield return ($"block{i}.mlp.fc2.weight", b.Mlp.Fc2.Weight);
            yield return ($"block{i}.mlp.fc2.bias", b.Mlp.Fc2.Bias!);
        }
        yield return ("lnf.gamma", _lnF.Gamma);
        yield return ("lnf.beta", _lnF.Beta);
        yield return ("lmhead.weight", _lmHead.Weight);
    }

    public override IEnumerable<Tensor> Parameters()
    {
        foreach (var p in _embedding.Parameters()) yield return p;
        foreach (var block in _blocks)
            foreach (var p in block.Parameters()) yield return p;
        foreach (var p in _lnF.Parameters()) yield return p;
        foreach (var p in _lmHead.Parameters()) yield return p;
    }
}
