namespace LLMExploration.Core.Layers;

/// <summary>
/// Base class for all neural network layers (modules).
///
/// Every layer that has trainable parameters inherits from Module.
/// The Parameters() method returns all leaf tensors that need gradients,
/// which the optimizer uses to update weights.
///
/// The training loop looks like:
///   1. optimizer.ZeroGrad()  (or module.ZeroGrad())
///   2. logits = model.Forward(tokens)
///   3. loss = CrossEntropy(logits, targets)
///   4. model.Backward(logits, targets)
///   5. optimizer.Step()
/// </summary>
public abstract class Module
{
    /// <summary>
    /// Return all trainable parameter tensors in this module (and its children).
    /// Each returned tensor must have RequiresGrad=true and a Grad buffer.
    /// </summary>
    public abstract IEnumerable<Tensor> Parameters();

    /// <summary>
    /// Zero the gradient buffers of all parameters.
    /// Must be called before each forward/backward pass.
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var p in Parameters())
            p.ZeroGrad();
    }

    /// <summary>
    /// Count total trainable parameters. Useful for understanding model scale.
    /// </summary>
    public int ParameterCount()
    {
        int count = 0;
        foreach (var p in Parameters())
            count += p.Size;
        return count;
    }
}
