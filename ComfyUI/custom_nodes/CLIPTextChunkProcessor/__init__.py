from .clip_text_processor import CLIPTextChunkProcessorNode

NODE_CLASS_MAPPINGS = {
    "CLIPTextChunkProcessorNode": CLIPTextChunkProcessorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextChunkProcessorNode": "CLIP Text Chunk Processor"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### Loading: CLIPTextChunkProcessorNode ###")
