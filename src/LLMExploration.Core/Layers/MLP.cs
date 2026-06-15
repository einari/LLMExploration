namespace LLMExploration.Core.Layers;

/// <summary>
/// Multi-Layer Perceptron (Feed-Forward Network) used within each transformer block.
///
/// After attention allows tokens to communicate, the MLP processes each token
/// independently, applying a non-linear transformation. This is where the model
/// "thinks" about what it learned from attending.
///
/// Architecture:
///   fc1: Linear(C, 4*C)    - expand to 4x width
///   gelu: GELU activation   - non-linearity
///   fc2: Linear(4*C, C)    - compress back to C
///
/// The 4x expansion ratio is standard in GPT. The MLP has ~8C^2 parameters
/// vs the attention's ~4C^2, so MLP dominates in large models.
///
/// GELU (Gaussian Error Linear Unit):
///   gelu(x) = x * Phi(x)  where Phi is the Gaussian CDF
///   Approximated as: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///   This is smoother than ReLU and empirically works better for transformers.
/// </summary>
public class MLP : Module
{
    public Linear Fc1 { get; }
    public Linear Fc2 { get; }

    private readonly int _nEmbd;

    // Cache for backward pass
    private Tensor? _fc1Output;   // pre-GELU output
    private Tensor? _geluOutput;  // post-GELU output

    // GELU constants
    private const float SqrtTwoDivPi = 0.7978845608028654f; // sqrt(2/pi)
    private const float GELUCoeff = 0.044715f;

    public MLP(int nEmbd, System.Random? rng = null)
    {
        _nEmbd = nEmbd;
        Fc1 = new Linear(nEmbd, 4 * nEmbd, useBias: true, rng: rng);
        Fc2 = new Linear(4 * nEmbd, nEmbd, useBias: true, rng: rng);
    }

    /// <summary>
    /// GELU activation function.
    /// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    ///
    /// Unlike ReLU which is piecewise linear, GELU is smooth and differentiable
    /// everywhere. The tanh approximation is standard in GPT-2.
    /// </summary>
    private static float Gelu(float x)
    {
        float inner = SqrtTwoDivPi * (x + GELUCoeff * x * x * x);
        return 0.5f * x * (1f + MathF.Tanh(inner));
    }

    /// <summary>
    /// Derivative of GELU: d/dx gelu(x)
    ///
    /// Using the product rule on gelu(x) = 0.5 * x * (1 + tanh(inner)):
    ///   d_inner/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
    ///   tanh_val   = tanh(inner)
    ///   sech2      = 1 - tanh_val^2  (derivative of tanh)
    ///   dgelu/dx   = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * d_inner
    /// </summary>
    private static float GeluGrad(float x)
    {
        float inner = SqrtTwoDivPi * (x + GELUCoeff * x * x * x);
        float tanhVal = MathF.Tanh(inner);
        float sech2 = 1f - tanhVal * tanhVal;
        float dInner = SqrtTwoDivPi * (1f + 3f * GELUCoeff * x * x);
        return 0.5f * (1f + tanhVal) + 0.5f * x * sech2 * dInner;
    }

    /// <summary>
    /// Forward pass: x -> fc1 -> gelu -> fc2
    /// Input/output shape: [B, T, C]
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        // fc1: [B, T, C] -> [B, T, 4C]
        _fc1Output = Fc1.Forward(x);

        // GELU: element-wise non-linearity
        _geluOutput = new Tensor(_fc1Output.Shape);
        for (int i = 0; i < _fc1Output.Size; i++)
            _geluOutput.Data[i] = Gelu(_fc1Output.Data[i]);

        // fc2: [B, T, 4C] -> [B, T, C]
        return Fc2.Forward(_geluOutput);
    }

    /// <summary>
    /// Backward pass: chain rule through fc2, gelu, fc1.
    ///
    ///   dGeluOutput = Fc2.Backward(dOut)
    ///   dFc1Output  = dGeluOutput * gelu'(fc1Output)   (element-wise)
    ///   dX          = Fc1.Backward(dFc1Output)
    /// </summary>
    public Tensor Backward(Tensor dOut)
    {
        if (_fc1Output == null || _geluOutput == null)
            throw new InvalidOperationException("Must call Forward before Backward");

        // Backward through fc2
        var dGeluOutput = Fc2.Backward(dOut);

        // Backward through GELU: multiply by GELU derivative
        var dFc1Output = new Tensor(_fc1Output.Shape);
        for (int i = 0; i < _fc1Output.Size; i++)
            dFc1Output.Data[i] = dGeluOutput.Data[i] * GeluGrad(_fc1Output.Data[i]);

        // Backward through fc1
        return Fc1.Backward(dFc1Output);
    }

    public override IEnumerable<Tensor> Parameters()
    {
        foreach (var p in Fc1.Parameters()) yield return p;
        foreach (var p in Fc2.Parameters()) yield return p;
    }
}
