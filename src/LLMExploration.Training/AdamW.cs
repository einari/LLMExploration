using LLMExploration.Core;

namespace LLMExploration.Training;

/// <summary>
/// AdamW optimizer (Adam with decoupled weight decay).
///
/// Adam (Adaptive Moment Estimation) maintains a running estimate of:
///   m[t] = first moment (mean) of gradients
///   v[t] = second moment (uncentered variance) of gradients
///
/// These are used to adaptively scale the learning rate for each parameter.
/// Parameters with consistently large gradients get smaller effective learning rates,
/// while parameters with small or noisy gradients get larger effective learning rates.
///
/// The "W" in AdamW refers to decoupled weight decay:
///   - Standard L2 regularization adds wd * param to the gradient
///   - AdamW applies weight decay directly to the parameter: param *= (1 - lr * wd)
///   - This avoids interaction with the adaptive learning rate, which is cleaner
///
/// Update rule:
///   m = beta1 * m + (1 - beta1) * g
///   v = beta2 * v + (1 - beta2) * g^2
///   m_hat = m / (1 - beta1^t)    (bias correction for initial steps)
///   v_hat = v / (1 - beta2^t)    (bias correction for initial steps)
///   param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
///
/// Typical hyperparameters (from the AdamW paper):
///   lr = 3e-4 (or use a schedule)
///   beta1 = 0.9    (heavy momentum on gradient mean)
///   beta2 = 0.999  (very heavy momentum on gradient variance)
///   eps = 1e-8     (prevents division by zero)
///   weight_decay = 0.1 (regularization strength)
///
/// Reference: "Decoupled Weight Decay Regularization" (Loshchilov and Hutter, 2017)
/// </summary>
public class AdamW
{
    private readonly float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;

    // First moment estimates (momentum): one float[] per parameter tensor
    private readonly float[][] _m;

    // Second moment estimates (velocity): one float[] per parameter tensor
    private readonly float[][] _v;

    // Parameter tensors (maintained in the same order as m and v)
    private readonly Tensor[] _params;

    // Step counter for bias correction
    private int _step;

    /// <summary>
    /// Create an AdamW optimizer for the given parameters.
    /// </summary>
    /// <param name="parameters">All trainable parameter tensors from the model.</param>
    /// <param name="lr">Learning rate (step size). Typical: 3e-4.</param>
    /// <param name="beta1">Decay rate for first moment. Typical: 0.9.</param>
    /// <param name="beta2">Decay rate for second moment. Typical: 0.999.</param>
    /// <param name="eps">Small constant for numerical stability. Typical: 1e-8.</param>
    /// <param name="weightDecay">L2 regularization coefficient. Typical: 0.1.</param>
    public AdamW(
        IEnumerable<Tensor> parameters,
        float lr = 3e-4f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0.1f)
    {
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;

        _params = parameters.ToArray();
        _m = _params.Select(p => new float[p.Size]).ToArray();
        _v = _params.Select(p => new float[p.Size]).ToArray();
        _step = 0;
    }

    /// <summary>
    /// Perform one optimization step.
    ///
    /// For each parameter tensor and each element:
    ///   1. Read the accumulated gradient from param.Grad
    ///   2. Update first and second moment estimates
    ///   3. Apply bias correction
    ///   4. Update the parameter
    ///
    /// Note: this does NOT zero gradients. Call model.ZeroGrad() before the next
    /// forward pass to reset the gradient accumulators.
    /// </summary>
    public void Step()
    {
        _step++;
        float t = _step;

        // Bias correction factors
        // Early in training (small t), m and v are biased toward zero
        // because they're initialized at 0. The correction factors counteract this.
        float bc1 = 1f - MathF.Pow(_beta1, t);
        float bc2 = 1f - MathF.Pow(_beta2, t);

        for (int pi = 0; pi < _params.Length; pi++)
        {
            var param = _params[pi];
            if (param.Grad == null) continue;

            float[] m = _m[pi];
            float[] v = _v[pi];
            float[] data = param.Data;
            float[] grad = param.Grad;

            for (int i = 0; i < param.Size; i++)
            {
                float g = grad[i];

                // Update biased first moment estimate (exponential moving average of gradients)
                m[i] = _beta1 * m[i] + (1f - _beta1) * g;

                // Update biased second raw moment estimate (EMA of squared gradients)
                v[i] = _beta2 * v[i] + (1f - _beta2) * g * g;

                // Compute bias-corrected estimates
                float mHat = m[i] / bc1;
                float vHat = v[i] / bc2;

                // AdamW update: Adam gradient step + decoupled weight decay
                // The weight decay term (- lr * wd * param) directly reduces weights
                // toward zero, acting as L2 regularization
                data[i] -= _lr * (mHat / (MathF.Sqrt(vHat) + _eps) + _weightDecay * data[i]);
            }
        }
    }
}
