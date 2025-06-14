# ComfyUI/custom_nodes/HelloWorldNode/nodes.py

class HelloWorld:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {
                    "multiline": False,
                    "default": "World"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("greeting_output",)

    FUNCTION = "greet"

    CATEGORY = "Example"

    def greet(self, name):
        if not name or name.strip() == "":
            name = "World"
        greeting = f"Hello, {name}!"
        print(f"HelloWorldNode: Generated greeting: {greeting}")
        return (greeting,)

# A dictionary that ComfyUI uses to know what nodes are available.
# This line is not strictly necessary here if it's already in __init__.py
# but can be useful for direct testing or if this file is used elsewhere.
# NODE_CLASS_MAPPINGS = {
#     "HelloWorld": HelloWorld
# }
