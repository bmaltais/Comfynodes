# ComfyUI Hello World Node

This repository contains a simple "Hello World" custom node for ComfyUI.

## Description

The "Hello World Node" is a basic example demonstrating how to create a custom node for ComfyUI. It takes a string input (a name) and outputs a greeting string.

## Installation

1.  Clone or download this repository.
2.  Place the `HelloWorldNode` directory into your `ComfyUI/custom_nodes/` directory.
3.  Restart ComfyUI.

You should then find the "Hello World Node" under the "Example" category in the node menu.

## Usage

1.  Add the "Hello World Node" to your graph.
2.  Optionally, provide a name in the "name" input field.
3.  Connect the `greeting_output` to another node that accepts a string (e.g., a text display node if available, or use it as input for another custom node).

The node will output "Hello, [name]!" or "Hello, World!" if no name is provided.
