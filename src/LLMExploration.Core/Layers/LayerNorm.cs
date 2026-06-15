namespace LLMExploration.Core.Layers;

/// <summary>
/// Layer Normalization: normalizes the last dimension of the input tensor.
///
/// Formula: Y = (X - mean) / sqrt(var + eps) * gamma + beta
///
/// Layer norm is applied independently to each token (position) and normalizes
/// across the feature (embedding) dimension. This is different from BatchNorm
/// which normalizes across the batch dimension.
///
/// Why layer norm?
///   - Stabilizes training by keeping activations in a well-scaled range
///   - Allows higher learning rates
///   - Independent of batch size (works for batch=1, unlike batch norm)
///
/// gamma and beta are learnable parameters that let the model "undo" the
/// normalization if that's optimal. They are initialized to gamma=1, beta=0
/// (identity transformation).
///
/// Input shape: [T, C] or [B, T, C] where C is the normalized dimension.
/// </summary>
public class LayerNorm : Module
{
    /// <summary>Learnable scale parameter. Shape: [NFeatures]. Initialized to 1.</summary>
    public Tensor Gamma { get; }

    /// <summary>Learnable shift parameter. Shape: [NFeatures]. Initialized to 0.</summary>
    public Tensor Beta { get; }

    private readonly int _nFeatures;
    private readonly float _eps;

    // Cache for backward pass
    private Tensor? _xhat;    // normalized input: (x - mean) / std
    private float[]? _rstd;   // 1/std for each token position
    private Tensor? _lastInput;
    private int _B, _T;

    public LayerNorm(int nFeatures, float eps = 1e-5f)
    {
        _nFeatures = nFeatures;
        _eps = eps;

        // gamma=1 so initially Y=Xhat (normalized), beta=0 so no shift
        Gamma = Tensor.Ones(new[] { nFeatures }, requiresGrad: true);
        Beta = new Tensor(new[] { nFeatures }, requiresGrad: true); // zeros
    }

    /// <summary>
    /// Forward pass: normalize each token's feature vector.
    ///
    /// For each token position (b, t):
    ///   mean = average of X[b,t,:]
    ///   var  = variance of X[b,t,:]
    ///   Xhat = (X - mean) / sqrt(var + eps)
    ///   Y    = Xhat * gamma + beta
    ///
    /// Caches Xhat and rstd=1/sqrt(var+eps) for backward pass.
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        _lastInput = x;
        int ndim = x.Shape.Length;
        int B, T;
        if (ndim == 2) { B = 1; T = x.Shape[0]; }
        else { B = x.Shape[0]; T = x.Shape[1]; }
        _B = B; _T = T;
        int C = _nFeatures;

        _xhat = new Tensor(x.Shape);
        _rstd = new float[B * T];

        var output = new Tensor(x.Shape);

        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int offset = (b * T + t) * C;

                // Compute mean over C features for this token
                float mean = 0f;
                for (int c = 0; c < C; c++)
                    mean += x.Data[offset + c];
                mean /= C;

                // Compute variance: E[(X - mean)^2]
                float variance = 0f;
                for (int c = 0; c < C; c++)
                {
                    float diff = x.Data[offset + c] - mean;
                    variance += diff * diff;
                }
                variance /= C;

                // rstd = 1 / sqrt(var + eps) - cached for backward
                float rstd = 1f / MathF.Sqrt(variance + _eps);
                _rstd[b * T + t] = rstd;

                // Compute Xhat and output
                for (int c = 0; c < C; c++)
                {
                    float xhat_val = (x.Data[offset + c] - mean) * rstd;
                    _xhat.Data[offset + c] = xhat_val;
                    output.Data[offset + c] = xhat_val * Gamma.Data[c] + Beta.Data[c];
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Backward pass for layer normalization.
    ///
    /// Given dL/dY (dOut), we need to compute:
    ///   dBeta[c]  = sum_{b,t} dOut[b,t,c]
    ///   dGamma[c] = sum_{b,t} dOut[b,t,c] * Xhat[b,t,c]
    ///
    /// For dX, we use the chain rule through the normalization:
    ///   dXhat = dOut * gamma                          (element-wise)
    ///   dX = (1/std) * (dXhat - mean_c(dXhat) - Xhat * mean_c(dXhat * Xhat))
    ///
    /// where mean_c denotes averaging over the C (feature) dimension.
    ///
    /// Derivation:
    ///   Y[c] = Xhat[c] * gamma[c] + beta[c]
    ///   Xhat[c] = (X[c] - mean) * rstd
    ///   mean = (1/C) * sum_c X[c]
    ///   var  = (1/C) * sum_c (X[c] - mean)^2
    ///
    /// Differentiating through all dependencies on X[c] (via mean and var)
    /// gives the formula above.
    /// </summary>
    public Tensor Backward(Tensor dOut)
    {
        if (_lastInput == null || _xhat == null || _rstd == null)
            throw new InvalidOperationException("Must call Forward before Backward");

        int B = _B, T = _T, C = _nFeatures;
        var dX = new Tensor(_lastInput.Shape);

        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int offset = (b * T + t) * C;
                float rstd = _rstd[b * T + t];

                // Step 1: accumulate dGamma and dBeta
                for (int c = 0; c < C; c++)
                {
                    float dy = dOut.Data[offset + c];
                    float xh = _xhat.Data[offset + c];
                    Beta.Grad![c] += dy;
                    Gamma.Grad![c] += dy * xh;
                }

                // Step 2: compute dXhat = dOut * gamma
                // Step 3: compute mean(dXhat) and mean(dXhat * Xhat) over C
                float sum_dxhat = 0f;
                float sum_dxhat_xhat = 0f;
                for (int c = 0; c < C; c++)
                {
                    float dxhat = dOut.Data[offset + c] * Gamma.Data[c];
                    sum_dxhat += dxhat;
                    sum_dxhat_xhat += dxhat * _xhat.Data[offset + c];
                }
                float mean_dxhat = sum_dxhat / C;
                float mean_dxhat_xhat = sum_dxhat_xhat / C;

                // Step 4: dX = rstd * (dXhat - mean_dxhat - Xhat * mean_dxhat_xhat)
                for (int c = 0; c < C; c++)
                {
                    float dxhat = dOut.Data[offset + c] * Gamma.Data[c];
                    dX.Data[offset + c] = rstd * (dxhat - mean_dxhat - _xhat.Data[offset + c] * mean_dxhat_xhat);
                }
            }
        }

        return dX;
    }

    public override IEnumerable<Tensor> Parameters()
    {
        yield return Gamma;
        yield return Beta;
    }
}
