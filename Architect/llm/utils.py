import ast


def extract_function_source_from_exec(code: str, function_name: str, namespace: dict = None) -> str:
    """
    Extract the source code of a function from a code string.

    Args:
        code: The code string containing the function
        function_name: The name of the function to extract
        namespace: Optional namespace dictionary

    Returns:
        str: The source code of the function
    """
    tree = ast.parse(code)  # Parse the entire code into an AST
    for node in ast.walk(tree):  # Traverse the AST
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(code, node)

    print(f"Function '{function_name}' not found or not a valid function.")
    return None