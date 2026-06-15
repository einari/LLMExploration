using System.Diagnostics;
using LLMExploration.Core;
using LLMExploration.Core.Layers;

namespace LLMExploration.Training;

/// <summary>
/// The training loop that orchestrates forward passes, backward passes,
/// and parameter updates.
///
/// The fundamental training loop for a language model:
///
///   for each step:
///     1. ZeroGrad: clear accumulated gradients from previous step
///     2. GetBatch: sample random windows of text
///     3. Forward: compute predictions (logits)
///     4. Loss: compute cross-entropy between predictions and targets
///     5. Backward: compute gradients w.r.t. all parameters
///     6. Step: update parameters using AdamW
///
/// Over many steps, the loss should decrease as the model learns to predict
/// the training text better and better.
///
/// Overfitting:
///   With a small dataset and enough steps, the model will memorize the training
///   text. That's fine for this educational demo - we want the model to generate
///   text that resembles the training data.
///
/// Note on batching:
///   This implementation processes one sequence per "step" (batch size = 1)
///   for clarity. To improve training speed, you'd accumulate gradients over
///   multiple sequences before each optimizer step (gradient accumulation).
/// </summary>
public class Trainer
{
    private readonly GPT _model;
    private readonly DataLoader _dataLoader;
    private readonly AdamW _optimizer;
    private readonly Tokenizer _tokenizer;

    public int Steps { get; set; } = 2000;
    public int EvalInterval { get; set; } = 100;
    public int SampleInterval { get; set; } = 500;
    public int CheckpointInterval { get; set; } = 500;
    public string CheckpointPath { get; set; } = "model.bin";
    public int SampleLength { get; set; } = 100;
    public float SampleTemperature { get; set; } = 0.8f;

    public Trainer(GPT model, DataLoader dataLoader, AdamW optimizer, Tokenizer tokenizer)
    {
        _model = model;
        _dataLoader = dataLoader;
        _optimizer = optimizer;
        _tokenizer = tokenizer;
    }

    /// <summary>
    /// Run the training loop for the configured number of steps.
    ///
    /// Each step:
    ///   1. Sample a random batch of token windows
    ///   2. For each sequence in the batch (currently 1):
    ///      - Zero gradients
    ///      - Forward pass to get loss
    ///      - Backward pass to compute gradients
    ///   3. Optimizer step to update weights
    ///   4. Periodically print loss and sample text
    /// </summary>
    public void Train()
    {
        Console.WriteLine($"\n=== Starting Training ===");
        Console.WriteLine($"Steps: {Steps}, Eval every {EvalInterval}, Sample every {SampleInterval}");
        Console.WriteLine();

        var sw = Stopwatch.StartNew();
        float lossAccum = 0f;
        int lossCount = 0;
        long tokensProcessed = 0;
        var lastTime = sw.Elapsed;

        for (int step = 1; step <= Steps; step++)
        {
            // Get one random training sequence
            var (inputs, targets) = _dataLoader.GetBatch(batchSize: 1);
            int[] inputTokens  = inputs[0];
            int[] targetTokens = targets[0];

            // Zero gradients from previous step
            _model.ZeroGrad();

            // Forward + loss computation
            var (_, loss) = _model.Loss(inputTokens, targetTokens);

            // Backward pass: compute gradients
            _model.Backward(targetTokens);

            // Update parameters
            _optimizer.Step();

            lossAccum += loss;
            lossCount++;
            tokensProcessed += inputTokens.Length;

            // Print training progress
            if (step % EvalInterval == 0)
            {
                float avgLoss = lossAccum / lossCount;
                var elapsed = sw.Elapsed;
                var dt = elapsed - lastTime;
                double tokensPerSec = tokensProcessed / dt.TotalSeconds;

                Console.WriteLine($"Step {step,5}/{Steps} | Loss: {avgLoss:F4} | " +
                                  $"Tokens/sec: {tokensPerSec:F0} | " +
                                  $"Elapsed: {elapsed.TotalSeconds:F1}s");

                lossAccum = 0f;
                lossCount = 0;
                tokensProcessed = 0;
                lastTime = elapsed;
            }

            // Print a sample generation to see model progress
            if (step % SampleInterval == 0)
            {
                PrintSample(step);
            }

            // Save checkpoint periodically
            if (step % CheckpointInterval == 0)
            {
                _model.Save(CheckpointPath);
                Console.WriteLine($"  [Checkpoint saved to {CheckpointPath}]");
            }
        }

        Console.WriteLine($"\nTraining complete! Total time: {sw.Elapsed.TotalSeconds:F1}s");
    }

    /// <summary>
    /// Generate and print a text sample to show model progress.
    /// Uses the first few characters of the training data as a seed.
    /// </summary>
    private void PrintSample(int step)
    {
        Console.WriteLine($"\n--- Sample at step {step} ---");

        // Seed with a few tokens from the training data
        var seedTokens = _dataLoader.GetTokens().Take(5).ToArray();
        string seedText = _tokenizer.Decode(seedTokens);
        Console.Write($"Seed: [{seedText}] -> ");

        try
        {
            string generated = _model.Generate(
                seedTokens, SampleLength, SampleTemperature, _tokenizer);
            Console.WriteLine(generated);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Generation failed: {ex.Message}]");
        }

        Console.WriteLine("---");
        Console.WriteLine();
    }
}
