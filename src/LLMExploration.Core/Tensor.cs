using System.Text;

namespace LLMExploration.Core;

/// <summary>
/// A CPU-only float tensor with shape tracking and optional gradient accumulation.
///
/// This is a simplified tensor that stores data in a flat float[] array using
/// row-major (C-style) layout. There is no autograd graph - gradients are
/// accumulated manually in each layer's Backward() method.
///
/// Shape example: shape=[2,3,4] means 2 batches, 3 tokens, 4 features.
/// Flat index: data[b*3*4 + t*4 + f] for element [b,t,f].
/// </summary>
public class Tensor
{
    /// <summary>Flat storage for all elements in row-major order.</summary>
    public float[] Data { get; }

    /// <summary>Dimension sizes, e.g. [batch, seq, features].</summary>
    public int[] Shape { get; }

    /// <summary>
    /// Gradient accumulation buffer. null if this tensor is not a parameter.
    /// After each backward pass, ZeroGrad() must be called before the next forward.
    /// </summary>
    public float[]? Grad { get; private set; }

    /// <summary>Whether gradients should be tracked for this tensor.</summary>
    public bool RequiresGrad { get; }

    /// <summary>Total number of elements (product of shape dimensions).</summary>
    public int Size { get; }

    public Tensor(int[] shape, bool requiresGrad = false)
    {
        Shape = shape;
        Size = 1;
        foreach (var d in shape) Size *= d;
        Data = new float[Size];
        RequiresGrad = requiresGrad;
        if (requiresGrad)
            Grad = new float[Size];
    }

    public Tensor(float[] data, int[] shape, bool requiresGrad = false)
    {
        Shape = shape;
        Size = 1;
        foreach (var d in shape) Size *= d;
        if (data.Length != Size)
            throw new ArgumentException($"Data length {data.Length} != shape product {Size}");
        Data = data;
        RequiresGrad = requiresGrad;
        if (requiresGrad)
            Grad = new float[Size];
    }

    /// <summary>
    /// Compute strides for each dimension in row-major order.
    /// stride[i] = product of shape[i+1..n-1].
    /// Used to convert multi-dimensional indices to flat index.
    /// </summary>
    public int[] Strides()
    {
        var strides = new int[Shape.Length];
        strides[Shape.Length - 1] = 1;
        for (int i = Shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * Shape[i + 1];
        return strides;
    }

    /// <summary>
    /// Convert multi-dimensional index array to flat index.
    /// Example: shape=[2,3,4], indices=[1,2,3] -> 1*12 + 2*4 + 3 = 23
    /// </summary>
    public int FlatIndex(int[] indices)
    {
        if (indices.Length != Shape.Length)
            throw new ArgumentException("Index rank must match shape rank");
        var strides = Strides();
        int idx = 0;
        for (int i = 0; i < indices.Length; i++)
            idx += indices[i] * strides[i];
        return idx;
    }

    /// <summary>2D element access by row and column (for 2D tensors).</summary>
    public float this[int i, int j]
    {
        get => Data[i * Shape[1] + j];
        set => Data[i * Shape[1] + j] = value;
    }

    /// <summary>3D element access (for [B,T,C] tensors).</summary>
    public float this[int i, int j, int k]
    {
        get => Data[i * Shape[1] * Shape[2] + j * Shape[2] + k];
        set => Data[i * Shape[1] * Shape[2] + j * Shape[2] + k] = value;
    }

    /// <summary>4D element access (for [B,H,T,D] attention tensors).</summary>
    public float this[int i, int j, int k, int l]
    {
        get => Data[i * Shape[1] * Shape[2] * Shape[3] + j * Shape[2] * Shape[3] + k * Shape[3] + l];
        set => Data[i * Shape[1] * Shape[2] * Shape[3] + j * Shape[2] * Shape[3] + k * Shape[3] + l] = value;
    }

    /// <summary>Reset gradient buffer to all zeros. Call before each forward pass.</summary>
    public void ZeroGrad()
    {
        if (Grad != null)
            Array.Clear(Grad, 0, Grad.Length);
    }

    /// <summary>Fill all data elements with a constant value.</summary>
    public void Fill(float v)
    {
        Array.Fill(Data, v);
    }

    /// <summary>Enable gradient tracking by allocating the grad buffer.</summary>
    public void EnableGrad()
    {
        if (Grad == null)
            Grad = new float[Size];
    }

    /// <summary>Create a tensor of all zeros with given shape.</summary>
    public static Tensor Zeros(int[] shape, bool requiresGrad = false)
    {
        return new Tensor(shape, requiresGrad);
    }

    /// <summary>Create a tensor of all ones with given shape.</summary>
    public static Tensor Ones(int[] shape, bool requiresGrad = false)
    {
        var t = new Tensor(shape, requiresGrad);
        Array.Fill(t.Data, 1.0f);
        return t;
    }

    /// <summary>
    /// Create a tensor with values sampled from Normal(mean, std).
    /// Uses Box-Muller transform for sampling.
    /// </summary>
    public static Tensor Random(int[] shape, float mean = 0f, float std = 0.02f,
        bool requiresGrad = false, System.Random? rng = null)
    {
        rng ??= new System.Random();
        var t = new Tensor(shape, requiresGrad);
        for (int i = 0; i < t.Size; i += 2)
        {
            // Box-Muller transform: converts two uniform randoms to two normal randoms
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double mag = std * Math.Sqrt(-2.0 * Math.Log(u1));
            t.Data[i] = (float)(mag * Math.Cos(2 * Math.PI * u2) + mean);
            if (i + 1 < t.Size)
                t.Data[i + 1] = (float)(mag * Math.Sin(2 * Math.PI * u2) + mean);
        }
        return t;
    }

    /// <summary>
    /// Extract a 2D slice [seqLen, features] from a 3D tensor [batch, seqLen, features].
    /// Used to extract a single batch element.
    /// </summary>
    public Tensor SliceBatch(int batchIdx)
    {
        if (Shape.Length != 3)
            throw new InvalidOperationException("SliceBatch requires 3D tensor");
        int T = Shape[1], C = Shape[2];
        var result = new Tensor(new[] { T, C });
        Array.Copy(Data, batchIdx * T * C, result.Data, 0, T * C);
        return result;
    }

    /// <summary>
    /// Copy data from a flat source array into this tensor starting at flatOffset.
    /// </summary>
    public void CopyFrom(float[] source, int sourceOffset, int destOffset, int count)
    {
        Array.Copy(source, sourceOffset, Data, destOffset, count);
    }

    /// <summary>
    /// Human-readable summary: shows shape and first few values.
    /// </summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append($"Tensor(shape=[{string.Join(",", Shape)}], ");
        int show = Math.Min(8, Size);
        sb.Append("data=[");
        for (int i = 0; i < show; i++)
        {
            sb.Append($"{Data[i]:F4}");
            if (i < show - 1) sb.Append(", ");
        }
        if (Size > show) sb.Append(", ...");
        sb.Append("])");
        return sb.ToString();
    }
}
