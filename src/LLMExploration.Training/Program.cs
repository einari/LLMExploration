using LLMExploration.Core;
using LLMExploration.Core.Layers;
using LLMExploration.Training;

Console.WriteLine("=================================================");
Console.WriteLine("  LLM Exploration - Training a Tiny GPT from Scratch");
Console.WriteLine("=================================================");
Console.WriteLine();

// ─── Step 1: Find or create training data ───────────────────────────────────

// Look for input.txt in current directory, parent directories, and known project paths
string[] searchPaths = {
    "data/input.txt",
    "../data/input.txt",
    "../../data/input.txt",
    Path.Combine(AppContext.BaseDirectory, "data/input.txt"),
    Path.Combine(AppContext.BaseDirectory, "../../../data/input.txt"),
    Path.Combine(AppContext.BaseDirectory, "../../../../data/input.txt"),
};

string? dataPath = null;
foreach (var candidate in searchPaths)
{
    if (File.Exists(candidate))
    {
        dataPath = Path.GetFullPath(candidate);
        break;
    }
}

if (dataPath == null)
{
    // Create a small sample text to train on
    dataPath = "data/input.txt";
    Directory.CreateDirectory("data");
    File.WriteAllText(dataPath, """
        Once upon a time there was a small model learning to speak.
        It read many words and tried to guess what came next.
        At first it was confused by all the letters and spaces.
        But slowly, step by step, it began to see patterns emerge.
        The model learned that certain letters follow others.
        It learned that words have rhythm and sentences have flow.
        And one day, it generated its very first coherent sentence.
        The researchers cheered and celebrated their tiny creation.
        For even a small mind that learns is a wonderful thing.
        And so the model kept training, kept improving, kept growing.
        """);
    Console.WriteLine($"[INFO] Created sample training data at '{dataPath}'");
}
else
{
    Console.WriteLine($"[INFO] Found training data at '{dataPath}'");
}

string trainingText = File.ReadAllText(dataPath);
Console.WriteLine($"[INFO] Training text: {trainingText.Length:N0} characters");
Console.WriteLine();

// ─── Step 2: Build tokenizer ────────────────────────────────────────────────

Console.WriteLine("Building tokenizer...");
var tokenizer = Tokenizer.Build(trainingText);
Console.WriteLine($"  Vocabulary size: {tokenizer.VocabSize} unique characters");

// Show the vocabulary
var allChars = trainingText.Distinct().OrderBy(c => c).ToArray();
Console.Write("  Characters: [");
foreach (var c in allChars)
    Console.Write(c == '\n' ? "\\n" : c == ' ' ? "_" : c.ToString());
Console.WriteLine("]");
Console.WriteLine();

// ─── Step 3: Configure the model ────────────────────────────────────────────

var config = new GPTConfig
{
    VocabSize = tokenizer.VocabSize,
    BlockSize = 64,    // context window: 64 characters
    NEmbedding = 64,   // embedding dimension
    NHead = 4,         // 4 attention heads (head dim = 16)
    NLayer = 4,        // 4 transformer blocks
};

Console.WriteLine("Model configuration:");
Console.WriteLine($"  VocabSize:  {config.VocabSize}");
Console.WriteLine($"  BlockSize:  {config.BlockSize} tokens");
Console.WriteLine($"  NEmbedding: {config.NEmbedding}");
Console.WriteLine($"  NHead:      {config.NHead} (head dim = {config.HeadDim})");
Console.WriteLine($"  NLayer:     {config.NLayer}");
Console.WriteLine();

// ─── Step 4: Create model ───────────────────────────────────────────────────

Console.WriteLine("Initializing GPT model...");
var rng = new Random(42);
var model = new GPT(config, rng);

int paramCount = model.ParameterCount();
Console.WriteLine($"  Total parameters: {paramCount:N0}");
Console.WriteLine($"  Memory estimate:  {paramCount * 4 / 1024.0 / 1024.0:F2} MB (weights only)");
Console.WriteLine();

// ─── Step 5: Create data loader ─────────────────────────────────────────────

var dataLoader = new DataLoader(dataPath, tokenizer, config.BlockSize, seed: 42);
Console.WriteLine();

// ─── Step 6: Create optimizer ───────────────────────────────────────────────

var optimizer = new AdamW(
    model.Parameters(),
    lr: 3e-4f,
    beta1: 0.9f,
    beta2: 0.999f,
    eps: 1e-8f,
    weightDecay: 0.1f
);
Console.WriteLine("Optimizer: AdamW (lr=3e-4, beta1=0.9, beta2=0.999, wd=0.1)");
Console.WriteLine();

// Quick sanity check: compute initial loss (should be ~log(VocabSize) for random weights)
Console.WriteLine("Sanity check - computing initial loss...");
var (_, initLoss) = model.Loss(
    dataLoader.GetTokens().Take(config.BlockSize).ToArray(),
    dataLoader.GetTokens().Skip(1).Take(config.BlockSize).ToArray()
);
float expectedLoss = MathF.Log(config.VocabSize);
Console.WriteLine($"  Initial loss: {initLoss:F4} (expected ~{expectedLoss:F4} = log({config.VocabSize}))");
Console.WriteLine();

// ─── Step 7: Train ──────────────────────────────────────────────────────────

var trainer = new Trainer(model, dataLoader, optimizer, tokenizer)
{
    Steps = 2000,
    EvalInterval = 100,
    SampleInterval = 500,
    CheckpointInterval = 500,
    CheckpointPath = "model.bin",
    SampleLength = 150,
    SampleTemperature = 0.8f,
};

trainer.Train();

// ─── Step 8: Save ───────────────────────────────────────────────────────────

Console.WriteLine("\nSaving model and tokenizer...");
model.Save("model.bin");
tokenizer.Save("tokenizer.json");
Console.WriteLine("  Saved model to 'model.bin'");
Console.WriteLine("  Saved tokenizer to 'tokenizer.json'");

// ─── Step 9: Final samples ──────────────────────────────────────────────────

Console.WriteLine("\n=== Final Generated Samples ===");
var sampleRng = new Random(123);

string[] seeds = { "Once", "The", "Finn" };
foreach (var seed in seeds)
{
    var seedTokens = tokenizer.Encode(seed);
    if (seedTokens.Length == 0) continue;

    Console.Write($"\nSeed: \"{seed}\" -> \"");
    string generated = model.Generate(seedTokens, 200, 0.8f, tokenizer, sampleRng);
    Console.Write(seed);
    Console.Write(generated);
    Console.WriteLine("\"");
}

Console.WriteLine();
Console.WriteLine("Training complete! Run LLMExploration.Chat to have a conversation.");
