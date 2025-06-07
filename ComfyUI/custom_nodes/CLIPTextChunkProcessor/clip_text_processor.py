import torch

class CLIPTextChunkProcessorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Enter text here..."
                }),
                # chunk_size is currently by words. For true token-based chunking,
                # this logic would need to use the CLIP tokenizer.
                "chunk_size": ("INT", {
                    "default": 75, # Target number of words per chunk.
                    "min": 10,
                    "max": 300,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("concatenated_conditioning",)
    FUNCTION = "process_text"
    CATEGORY = "Text Processing"

    def process_text(self, clip, text, chunk_size):
        print(f"CLIPTextChunkProcessorNode: Received text (first 100 chars): {text[:100]}...")
        print(f"CLIPTextChunkProcessorNode: Target words per chunk: {chunk_size}")

        if not text or not text.strip():
            print("CLIPTextChunkProcessorNode: Empty text received, returning None.")
            return (None,)

        words = text.split()

        if not words:
            print("CLIPTextChunkProcessorNode: Text contains no words after splitting, returning None.")
            return (None,)

        # Current chunking is by word count. This is a proxy for token count.
        # For more precise token-based chunking, one would typically:
        # 1. Tokenize the entire text using `clip.tokenize()`.
        # 2. Split the resulting token IDs into chunks of the desired token length.
        # 3. Convert these token ID chunks back to text strings (if `clip.encode` requires text)
        #    or use a method like `clip.encode_from_tokens()` if available.
        # This simplified version splits text strings by word count, then encodes those.
        text_chunks = []
        current_chunk_words = []
        for word in words:
            current_chunk_words.append(word)
            if len(current_chunk_words) >= chunk_size:
                text_chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []

        if current_chunk_words: # Add any remaining words as the last chunk
            text_chunks.append(" ".join(current_chunk_words))

        if not text_chunks:
            print("CLIPTextChunkProcessorNode: No text chunks were created, returning None.")
            return (None,)

        print(f"CLIPTextChunkProcessorNode: Number of text chunks created: {len(text_chunks)}")

        all_conditionings = []

        for i, chunk_text in enumerate(text_chunks):
            print(f"CLIPTextChunkProcessorNode: Processing chunk {i+1}/{len(text_chunks)} (first 100 chars): {chunk_text[:100]}...")
            try:
                # `clip.encode(text)` will tokenize the `chunk_text`.
                # If `chunk_text` tokenizes to more than the CLIP model's max sequence length (e.g., 77 tokens),
                # it will typically be truncated by the tokenizer within `clip.encode()`.
                encoded_output = clip.encode(chunk_text)

                if encoded_output and isinstance(encoded_output, list) and \
                   len(encoded_output) > 0 and isinstance(encoded_output[0], list) and \
                   len(encoded_output[0]) > 0 and torch.is_tensor(encoded_output[0][0]):
                    conditioning_tensor = encoded_output[0][0]
                    all_conditionings.append(conditioning_tensor)
                    # Actual number of tokens used by CLIP for this chunk might be available
                    # in `encoded_output[0][1]` (the metadata dict), e.g., under a 'length' key.
                    # Logging this would be useful for debugging chunking.
                    # For example: token_length = encoded_output[0][1].get('length', 'unknown')
                    # print(f"CLIPTextChunkProcessorNode: Chunk {i+1} encoded. Shape: {conditioning_tensor.shape}, Tokens: {token_length}")
                    print(f"CLIPTextChunkProcessorNode: Chunk {i+1} encoded successfully. Shape: {conditioning_tensor.shape}")
                else:
                    print(f"CLIPTextChunkProcessorNode: Error or unexpected output format from clip.encode for chunk {i+1}. Output was: {encoded_output}")

            except Exception as e:
                print(f"CLIPTextChunkProcessorNode: Error encoding chunk {i+1} with CLIP: {e}")
                # Depending on desired robustness, might skip failed chunks or abort.

        if not all_conditionings:
            print("CLIPTextChunkProcessorNode: No conditionings were generated from any chunks, returning None.")
            return (None,)

        try:
            # Concatenate along the sequence dimension (dim=1).
            # Assumes conditionings are [1, num_tokens_in_chunk, features].
            concatenated_conditioning = torch.cat(all_conditionings, dim=1)
            print(f"CLIPTextChunkProcessorNode: Concatenated conditioning shape: {concatenated_conditioning.shape}")

            return ([[concatenated_conditioning, {}]], )

        except Exception as e:
            print(f"CLIPTextChunkProcessorNode: Error concatenating conditionings: {e}")
            return (None,)
