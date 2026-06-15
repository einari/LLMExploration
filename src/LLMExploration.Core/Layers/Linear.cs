namespace LLMExploration.Core.Layers;

/// <summary>
/// Fully-connected (dense) linear layer: y = x @ W^T + b
///
/// This is the workhorse of the transformer. Every projection in attention
/// and the MLP uses this layer.
///
/// Weight shape: [outFeatures, inFeatures]  (transposed relative to math convention)
/// Bias shape:   [outFeatures]
///
/// For a batched input x of shape [B, T, inFeatures], the output is [B, T, outFeatures].
/// Each token is processed independently by the same weight matrix.
///
/// The weight is stored as [out, in] so that W[o, i] is accessed with W.Data[o*in + i],
/// which is cache-friendly when computing the dot product for output neuron o.
/// </summary>
public class Linear : Module
{
    /// <summary>Weight matrix. Shape: [OutFeatures, InFeatures].</summary>
    public Tensor Weight { get; }

    /// <summary>Bias vector. Shape: [OutFeatures]. Null if useBias=false.</summary>
    public Tensor? Bias { get; }

    public int InFeatures { get; }
    public int OutFeatures { get; }

    // Cache for backward pass
    private Tensor? _lastInput;

    public Linear(int inFeatures, int outFeatures, bool useBias = true, System.Random? rng = null)
    {
        InFeatures = inFeatures;
        OutFeatures = outFeatures;

        // GPT-2 initialization: Normal(0, 0.02) for all weights
        Weight = Tensor.Random(new[] { outFeatures, inFeatures }, std: 0.02f, requiresGrad: true, rng: rng);

        if (useBias)
        {
            Bias = new Tensor(new[] { outFeatures }, requiresGrad: true);
            // Bias starts at zero
        }
    }

    /// <summary>
    /// Forward pass: compute output = input @ Weight^T + bias
    ///
    /// For input shape [B, T, in]:
    ///   output[b, t, o] = sum_i input[b, t, i] * Weight[o, i] + Bias[o]
    ///
    /// This is equivalent to applying the same linear transformation to each
    /// (batch, token) position independently.
    /// </summary>
    /// <param name="x">Input tensor. Must end with dimension InFeatures.
    ///   Supported shapes: [in], [T, in], [B, T, in]</param>
    /// <returns>Output tensor with last dimension replaced by OutFeatures.</returns>
    public Tensor Forward(Tensor x)
    {
        _lastInput = x;

        // Determine shape
        int ndim = x.Shape.Length;
        int B, T;
        if (ndim == 1)
        {
            B = 1; T = 1;
        }
        else if (ndim == 2)
        {
            B = 1; T = x.Shape[0];
        }
        else
        {
            B = x.Shape[0]; T = x.Shape[1];
        }
        int inF = x.Shape[ndim - 1];
        if (inF != InFeatures)
            throw new ArgumentException($"Expected last dim {InFeatures}, got {inF}");

        // Output shape mirrors input shape but last dim = OutFeatures
        int[] outShape = ndim == 1
            ? new[] { OutFeatures }
            : ndim == 2
                ? new[] { T, OutFeatures }
                : new[] { B, T, OutFeatures };

        var output = new Tensor(outShape);

        // Core matmul: for each (b, t), compute dot products with each output neuron
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int xOffset = (b * T + t) * InFeatures;
                int yOffset = (b * T + t) * OutFeatures;

                for (int o = 0; o < OutFeatures; o++)
                {
                    float sum = Bias != null ? Bias.Data[o] : 0f;
                    int wOffset = o * InFeatures;
                    for (int i = 0; i < InFeatures; i++)
                    {
                        sum += x.Data[xOffset + i] * Weight.Data[wOffset + i];
                    }
                    output.Data[yOffset + o] = sum;
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Backward pass: given gradient dOut w.r.t. output, compute:
    ///   dX      = dOut @ Weight          (gradient w.r.t. input)
    ///   dWeight += sum_{b,t} dOut[b,t]^T outer x[b,t]  (gradient w.r.t. weights)
    ///   dBias   += sum_{b,t} dOut[b,t]                  (gradient w.r.t. bias)
    ///
    /// Note: weight and bias gradients are accumulated (+=) since this layer
    /// might be called multiple times (e.g., in unrolled loops).
    /// </summary>
    /// <param name="dOut">Gradient of loss w.r.t. output. Same shape as Forward output.</param>
    /// <returns>Gradient of loss w.r.t. input x. Same shape as x.</returns>
    public Tensor Backward(Tensor dOut)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Must call Forward before Backward");

        var x = _lastInput;
        int ndim = x.Shape.Length;
        int B, T;
        if (ndim == 1) { B = 1; T = 1; }
        else if (ndim == 2) { B = 1; T = x.Shape[0]; }
        else { B = x.Shape[0]; T = x.Shape[1]; }

        var dX = new Tensor(x.Shape);

        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int xOffset = (b * T + t) * InFeatures;
                int dyOffset = (b * T + t) * OutFeatures;

                for (int o = 0; o < OutFeatures; o++)
                {
                    float dout_bt_o = dOut.Data[dyOffset + o];

                    // dBias[o] += dOut[b,t,o]
                    if (Bias?.Grad != null)
                        Bias.Grad[o] += dout_bt_o;

                    int wOffset = o * InFeatures;
                    for (int i = 0; i < InFeatures; i++)
                    {
                        // dWeight[o, i] += dOut[b,t,o] * x[b,t,i]
                        if (Weight.Grad != null)
                            Weight.Grad[wOffset + i] += dout_bt_o * x.Data[xOffset + i];

                        // dX[b,t,i] += dOut[b,t,o] * Weight[o,i]
                        dX.Data[xOffset + i] += dout_bt_o * Weight.Data[wOffset + i];
                    }
                }
            }
        }

        return dX;
    }

    public override IEnumerable<Tensor> Parameters()
    {
        yield return Weight;
        if (Bias != null)
            yield return Bias;
    }
}
