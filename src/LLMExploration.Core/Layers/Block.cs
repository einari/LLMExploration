namespace LLMExploration.Core.Layers;

/// <summary>
/// A single Transformer Block: the repeating unit of the GPT architecture.
///
/// Each block alternates between two operations:
///   1. Attention (communication): tokens exchange information with each other
///   2. MLP (computation): each token processes its own representation independently
///
/// Both sub-layers use residual connections: x = x + sublayer(LayerNorm(x))
///
/// Why residual connections?
///   - Allow gradients to flow directly to earlier layers (avoid vanishing gradients)
///   - Let the model learn "residual" functions: how much to update the representation
///   - At initialization, the blocks are near-identity: output ≈ input
///
/// Why LayerNorm before each sublayer (Pre-LN)?
///   - Original paper used Post-LN (after), but Pre-LN (before) trains more stably
///   - GPT-2 and most modern transformers use Pre-LN
///   - Keeps the residual stream "clean" (unnormalized)
///
/// Block forward:
///   x = x + attn(ln1(x))   # attend, then add back to residual stream
///   x = x + mlp(ln2(x))    # compute, then add back to residual stream
/// </summary>
public class Block : Module
{
    public LayerNorm Ln1 { get; }
    public CausalSelfAttention Attn { get; }
    public LayerNorm Ln2 { get; }
    public MLP Mlp { get; }

    // Cache for backward pass (residual stream values)
    private Tensor? _xInput;        // x before block
    private Tensor? _ln1Out;        // output of ln1
    private Tensor? _attnOut;       // output of attention
    private Tensor? _xAfterAttn;    // x + attnOut (residual)
    private Tensor? _ln2Out;        // output of ln2
    private Tensor? _mlpOut;        // output of mlp

    public Block(int nEmbd, int nHead, System.Random? rng = null)
    {
        Ln1 = new LayerNorm(nEmbd);
        Attn = new CausalSelfAttention(nEmbd, nHead, rng);
        Ln2 = new LayerNorm(nEmbd);
        Mlp = new MLP(nEmbd, rng);
    }

    /// <summary>
    /// Forward pass through one transformer block.
    /// Input x: [B, T, C]
    /// Output:  [B, T, C]
    ///
    /// x = x + attn(ln1(x))
    /// x = x + mlp(ln2(x))
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        _xInput = x;

        // Attention sub-layer with residual connection
        _ln1Out = Ln1.Forward(x);
        _attnOut = Attn.Forward(_ln1Out);

        // Residual: add attention output back to input
        _xAfterAttn = new Tensor(x.Shape);
        for (int i = 0; i < x.Size; i++)
            _xAfterAttn.Data[i] = x.Data[i] + _attnOut.Data[i];

        // MLP sub-layer with residual connection
        _ln2Out = Ln2.Forward(_xAfterAttn);
        _mlpOut = Mlp.Forward(_ln2Out);

        // Residual: add MLP output back
        var output = new Tensor(x.Shape);
        for (int i = 0; i < x.Size; i++)
            output.Data[i] = _xAfterAttn.Data[i] + _mlpOut.Data[i];

        return output;
    }

    /// <summary>
    /// Backward pass through one transformer block.
    ///
    /// Given dOut (gradient w.r.t. block output), we trace back:
    ///
    /// Forward: out = xAfterAttn + mlp(ln2(xAfterAttn))
    /// Backward:
    ///   dXAfterAttn_from_mlp = Mlp.Backward(Ln2.Backward(dOut)) (through mlp branch)
    ///   dXAfterAttn = dOut + dXAfterAttn_from_mlp  (residual: gradient flows through +)
    ///
    /// Forward: xAfterAttn = x + attn(ln1(x))
    /// Backward:
    ///   dX_from_attn = Attn.Backward(Ln1.Backward(dXAfterAttn)) (through attn branch)
    ///   dX = dXAfterAttn + dX_from_attn  (residual)
    /// </summary>
    public Tensor Backward(Tensor dOut)
    {
        if (_xAfterAttn == null || _ln1Out == null || _ln2Out == null)
            throw new InvalidOperationException("Must call Forward before Backward");

        // Backward through MLP residual branch
        // out = xAfterAttn + mlpOut, so dXAfterAttn gets dOut directly (residual)
        // plus the gradient that flows through the MLP branch
        var dLn2Out = Mlp.Backward(dOut);
        var dXAfterAttn_mlp = Ln2.Backward(dLn2Out);

        // Combine: gradient flows through both the skip connection and the MLP branch
        var dXAfterAttn = new Tensor(_xAfterAttn.Shape);
        for (int i = 0; i < dXAfterAttn.Size; i++)
            dXAfterAttn.Data[i] = dOut.Data[i] + dXAfterAttn_mlp.Data[i];

        // Backward through attention residual branch
        var dLn1Out = Attn.Backward(dXAfterAttn);
        var dX_attn = Ln1.Backward(dLn1Out);

        // Combine: gradient through skip connection + attention branch
        var dX = new Tensor(_xInput!.Shape);
        for (int i = 0; i < dX.Size; i++)
            dX.Data[i] = dXAfterAttn.Data[i] + dX_attn.Data[i];

        return dX;
    }

    public override IEnumerable<Tensor> Parameters()
    {
        foreach (var p in Ln1.Parameters()) yield return p;
        foreach (var p in Attn.Parameters()) yield return p;
        foreach (var p in Ln2.Parameters()) yield return p;
        foreach (var p in Mlp.Parameters()) yield return p;
    }
}
