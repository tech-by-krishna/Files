# ML Kit Helpers for Mendix

Java actions that fill the gaps around Mendix ML Kit when running ONNX models, especially text models (encoder-decoder such as T5, or encoder-only). They handle tokenizing, tensor encoding and decoding, and next-token selection, so a full text-generation loop can be built in a microflow without external services.

> Not affiliated with Mendix. Do not bundle model or tokenizer files. Check each model's license before use (some, such as LaMini, are non-commercial).

## Requirements

- Mendix Studio Pro 10.x or later, with the Mendix ML Kit module installed
- DJL `0.36.0` HuggingFace tokenizer jars (included in `userlib`)
- A `tokenizer.json` for your model, placed in the app's `resources` folder

## Conventions

- All tensors travel as **base64 strings**. `int64` tensors use 8 bytes per value and `float32` use 4 bytes.
- Default byte order is **big-endian**. This is what Mendix ML Kit mappings accepted in testing. Actions with a `LittleEndian` parameter let you override it.
- Mendix `Integer/Long` parameters arrive as `Long` in Java. The actions convert internally.
- Sequence lengths must match the dimensions you set in the ML Kit mapping (for example `1, 256` for encoder inputs).

## Quick start: text generation (encoder-decoder)

1. `TokenizeText` on the prompt, which gives `input_ids` and `attention_mask`.
2. Call ML model (encoder), then keep `last_hidden_state`.
3. Create variables: `GeneratedTokenIds = "0"`, `CurrentLength = 1`, `LastToken = 0`.
4. While `$CurrentLength < MaxSteps and $LastToken != 1`:
   1. `TokenIdsCsvToBase64($GeneratedTokenIds, decoderLen)`
   2. Create the decoder input object. Set `past_key_values.*` to `ZerosToBase64`, `use_cache_branch = false`.
   3. Call ML model (decoder)
   4. `ArgmaxFromBase64` (or `ArgmaxWithRepetitionPenalty`)
   5. Append the token to `GeneratedTokenIds`, increment `CurrentLength`, set `LastToken`.
5. `substring($GeneratedTokenIds, 2)` to drop the start token, then `TokenIdsCsvToBase64`, then `DetokenizeText`.

## Actions

### TokenizeText
Converts text to token ids and an attention mask, truncated and padded to a fixed length.
- **Parameters:** text, max length, tokenizer path. *(Confirm against your final signature.)*
- **Returns:** `input_ids` (base64 int64) and attention mask (base64 int64).
- **Notes:** ends with the end-of-sequence token (`1` for T5). Output size is always `maxLength x 8` bytes.

### DetokenizeText
Converts token ids back to text.
- **Parameters:** base64 int64 token ids.
- **Returns:** text.
- **Notes:** stops at the first `0` (padding) or `1` (end token). Remove the leading start token `0` before calling it.

### TokenIdsCsvToBase64
Turns a comma-separated token list into a padded int64 tensor.
- **Parameters:** `IdsCsv` (String, for example `"0,363,19"`), `MaxLen` (Integer).
- **Returns:** base64 string of `MaxLen x 8` bytes, padded with `0`.
- **Notes:** `MaxLen` must equal the mapping's `input_ids` dimension.

### ArgmaxFromBase64
Picks the highest-scoring token for the current position.
- **Parameters:** `LogitsBase64`, `CurrentLength` (number of real tokens so far), `VocabSize` (for example `32128` for T5).
- **Returns:** token id (Long).
- **Notes:** reads only the slice for position `CurrentLength - 1`. Call it before incrementing `CurrentLength`.

### ZerosToBase64
Returns a base64 string of zero bytes.
- **Parameters:** `ByteCount`.
- **Returns:** base64 string.
- **Notes:** for merged decoders, set every `past_key_values.*` input to `ZerosToBase64(1536)` for shape `1,6,1,64` (float32) and `use_cache_branch` to `false`. Create the value once before the loop.

### SampleFromBase64 (planned)
Top-k and temperature sampling instead of greedy decoding.
- **Parameters:** `LogitsBase64`, `CurrentLength`, `VocabSize`, `TopK`, `Temperature`, optional seed.
- **Returns:** token id.

### Byte-order option (planned)
A `LittleEndian` Boolean on every tensor action, for models or runtimes that do not use big-endian.

### Tokenizer caching (internal)
The tokenizer is loaded once and reused across calls. No microflow change is needed.

### Also recommended
- `ArgmaxWithRepetitionPenalty`: adds `GeneratedTokenIds` and a penalty to reduce repeated phrases.
- `CountTokens` and `TruncateToTokens`: keep prompts (for example RAG context) within the fixed encoder length.

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `Unsupported MendixValue null` | An input attribute is empty. Check all `past_key_values.*` attributes. |
| `Illegal base64 character 28` | Attribute holds text such as `(`, not base64. Use a variable from the Java action. |
| `Shape [1, 64] requires 64 elements but the buffer has 256` | The `MaxLen` passed to `TokenIdsCsvToBase64` does not match the mapping's dimension. |
| Empty reply | A leading `0` was passed to `DetokenizeText`. Strip the start token. |
| Repeated phrases | Greedy decoding. Use the repetition-penalty or sampling action. |

## Limitations

- Input lengths are fixed by the ML Kit mapping, not dynamic.
- Without a key-value cache every step re-runs the whole decoder, so replies get slower as they grow. Very small models give weak answers unless you add context (RAG).
