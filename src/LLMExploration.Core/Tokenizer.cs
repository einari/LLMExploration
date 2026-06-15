using System.Text.Json;

namespace LLMExploration.Core;

/// <summary>
/// Character-level tokenizer: each unique character in the training text
/// becomes a token. This is the simplest possible tokenization scheme and
/// lets the model learn directly from raw characters.
///
/// Real LLMs use subword tokenizers (BPE, SentencePiece) with vocabularies
/// of 32k-100k tokens, but character-level is easier to understand and implement.
///
/// Usage:
///   var tok = Tokenizer.Build(text);
///   int[] ids = tok.Encode("hello");
///   string s   = tok.Decode(ids);
/// </summary>
public class Tokenizer
{
    /// <summary>Maps character -> integer token id.</summary>
    private readonly Dictionary<char, int> _charToId;

    /// <summary>Maps integer token id -> character.</summary>
    private readonly Dictionary<int, char> _idToChar;

    /// <summary>Number of unique tokens (= number of unique characters in training text).</summary>
    public int VocabSize => _charToId.Count;

    private Tokenizer(Dictionary<char, int> charToId, Dictionary<int, char> idToChar)
    {
        _charToId = charToId;
        _idToChar = idToChar;
    }

    /// <summary>
    /// Build a tokenizer by scanning the text and assigning each unique character
    /// an integer id in sorted order (so the mapping is deterministic).
    /// </summary>
    public static Tokenizer Build(string text)
    {
        // Collect all unique characters and sort them for a stable ordering
        var chars = text.Distinct().OrderBy(c => c).ToArray();
        var charToId = new Dictionary<char, int>();
        var idToChar = new Dictionary<int, char>();
        for (int i = 0; i < chars.Length; i++)
        {
            charToId[chars[i]] = i;
            idToChar[i] = chars[i];
        }
        return new Tokenizer(charToId, idToChar);
    }

    /// <summary>
    /// Convert a string to a sequence of integer token ids.
    /// Characters not in the vocabulary are skipped with a warning.
    /// </summary>
    public int[] Encode(string text)
    {
        var ids = new List<int>(text.Length);
        foreach (char c in text)
        {
            if (_charToId.TryGetValue(c, out int id))
                ids.Add(id);
            // Unknown chars are silently skipped
        }
        return ids.ToArray();
    }

    /// <summary>
    /// Convert a sequence of token ids back to a string.
    /// Unknown ids produce a '?' character.
    /// </summary>
    public string Decode(int[] tokens)
    {
        var sb = new System.Text.StringBuilder(tokens.Length);
        foreach (int id in tokens)
        {
            if (_idToChar.TryGetValue(id, out char c))
                sb.Append(c);
            else
                sb.Append('?');
        }
        return sb.ToString();
    }

    /// <summary>
    /// Serialize the vocabulary to a JSON file so it can be reloaded later
    /// without access to the original training text.
    /// </summary>
    public void Save(string path)
    {
        // Convert char keys to strings for JSON serialization
        var data = new TokenizerData
        {
            CharToId = _charToId.ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
        };
        var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    /// <summary>Load a previously saved tokenizer from a JSON file.</summary>
    public static Tokenizer Load(string path)
    {
        var json = File.ReadAllText(path);
        var data = JsonSerializer.Deserialize<TokenizerData>(json)
            ?? throw new InvalidDataException("Failed to parse tokenizer JSON");

        var charToId = data.CharToId.ToDictionary(kv => kv.Key[0], kv => kv.Value);
        var idToChar = charToId.ToDictionary(kv => kv.Value, kv => kv.Key);
        return new Tokenizer(charToId, idToChar);
    }

    // Helper class for JSON serialization
    private class TokenizerData
    {
        public Dictionary<string, int> CharToId { get; set; } = new();
    }
}
