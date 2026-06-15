using LLMExploration.Core;
using LLMExploration.Core.Layers;

Console.WriteLine("=================================================");
Console.WriteLine("  LLM Exploration - Interactive Chat");
Console.WriteLine("=================================================");
Console.WriteLine();

// ─── Step 1: Locate model files ────────────────────────────────────────────

string[] searchDirs = {
    ".",
    "..",
    "../..",
    AppContext.BaseDirectory,
    Path.Combine(AppContext.BaseDirectory, "../../.."),
    Path.Combine(AppContext.BaseDirectory, "../../../.."),
};

string? modelPath = null;
string? tokenizerPath = null;

foreach (var dir in searchDirs)
{
    string mp = Path.Combine(dir, "model.bin");
    string tp = Path.Combine(dir, "tokenizer.json");
    if (File.Exists(mp) && File.Exists(tp))
    {
        modelPath = Path.GetFullPath(mp);
        tokenizerPath = Path.GetFullPath(tp);
        break;
    }
}

if (modelPath == null || tokenizerPath == null)
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine("ERROR: Could not find model.bin and tokenizer.json");
    Console.ResetColor();
    Console.WriteLine();
    Console.WriteLine("Please run LLMExploration.Training first to train and save the model.");
    Console.WriteLine("Then run this program from the same directory.");
    Console.WriteLine();
    Console.Write("Press any key to exit...");
    Console.ReadKey();
    return;
}

Console.WriteLine($"Loading model from: {modelPath}");
Console.WriteLine($"Loading tokenizer from: {tokenizerPath}");
Console.WriteLine();

// ─── Step 2: Load model and tokenizer ──────────────────────────────────────

Tokenizer tokenizer;
GPT model;

try
{
    tokenizer = Tokenizer.Load(tokenizerPath);
    Console.WriteLine($"Tokenizer loaded: {tokenizer.VocabSize} unique characters in vocabulary");

    model = GPT.Load(modelPath);
    Console.WriteLine($"Model loaded: {model.ParameterCount():N0} parameters");
}
catch (Exception ex)
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"ERROR loading model: {ex.Message}");
    Console.ResetColor();
    Console.Write("Press any key to exit...");
    Console.ReadKey();
    return;
}

// ─── Step 3: Welcome message ────────────────────────────────────────────────

Console.WriteLine();
Console.WriteLine("About this model:");
Console.WriteLine("  This is a character-level GPT trained from scratch on a small text.");
Console.WriteLine("  It predicts one character at a time based on what came before.");
Console.WriteLine("  The output will resemble the style and content of the training text.");
Console.WriteLine();
Console.WriteLine("Commands:");
Console.WriteLine("  /temp <value>    Set temperature (default: 0.8, range: 0.1-2.0)");
Console.WriteLine("  /tokens <n>      Set max tokens to generate (default: 200)");
Console.WriteLine("  /help            Show this help");
Console.WriteLine("  /quit or /exit   Exit the program");
Console.WriteLine();
Console.WriteLine("Type any text as a prompt and the model will continue it.");
Console.WriteLine("─────────────────────────────────────────────────────────");
Console.WriteLine();

// ─── Step 4: Interactive loop ───────────────────────────────────────────────

float temperature = 0.8f;
int maxTokens = 200;
var rng = new Random();

while (true)
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write("You: ");
    Console.ResetColor();

    string? input = Console.ReadLine();
    if (input == null) break;

    input = input.Trim();
    if (string.IsNullOrEmpty(input)) continue;

    // Handle commands
    if (input.StartsWith("/"))
    {
        var parts = input.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
        string cmd = parts[0].ToLower();

        switch (cmd)
        {
            case "/quit":
            case "/exit":
                Console.WriteLine("Goodbye!");
                return;

            case "/help":
                Console.WriteLine("Commands:");
                Console.WriteLine("  /temp <value>    Set temperature (current: " + temperature + ")");
                Console.WriteLine("  /tokens <n>      Set max tokens (current: " + maxTokens + ")");
                Console.WriteLine("  /quit            Exit");
                break;

            case "/temp":
                if (parts.Length > 1 && float.TryParse(parts[1], out float newTemp) && newTemp > 0)
                {
                    temperature = Math.Clamp(newTemp, 0.01f, 10f);
                    Console.WriteLine($"Temperature set to {temperature:F2}");
                    Console.WriteLine("  Lower = more predictable, Higher = more creative/random");
                }
                else
                {
                    Console.WriteLine($"Usage: /temp <value>  (current: {temperature:F2})");
                }
                break;

            case "/tokens":
                if (parts.Length > 1 && int.TryParse(parts[1], out int newTokens) && newTokens > 0)
                {
                    maxTokens = Math.Clamp(newTokens, 1, 1000);
                    Console.WriteLine($"Max tokens set to {maxTokens}");
                }
                else
                {
                    Console.WriteLine($"Usage: /tokens <n>  (current: {maxTokens})");
                }
                break;

            default:
                Console.WriteLine($"Unknown command: {cmd}. Type /help for available commands.");
                break;
        }

        Console.WriteLine();
        continue;
    }

    // Encode the prompt
    var promptTokens = tokenizer.Encode(input);
    if (promptTokens.Length == 0)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("[Warning: Your input contains no characters known to this model's vocabulary.]");
        Console.WriteLine("[The model was trained on a specific text - try characters from that text.]");
        Console.ResetColor();
        Console.WriteLine();
        continue;
    }

    // Generate response
    Console.ForegroundColor = ConsoleColor.Green;
    Console.Write("Model: ");
    Console.ResetColor();

    // Echo the prompt
    Console.Write(input);

    try
    {
        string response = model.Generate(promptTokens, maxTokens, temperature, tokenizer, rng);
        Console.WriteLine(response);
    }
    catch (Exception ex)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"\n[Generation error: {ex.Message}]");
        Console.ResetColor();
    }

    Console.WriteLine();
}
