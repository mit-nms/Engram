import os
import re
from pathlib import Path
from typing import Any, Callable
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.protocol import WriteResult
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import AgentMiddleware


def add_file_to_agent_filesystem(
    file_path: str,
    backend: BackendProtocol,
    agent_path: str,
    tool_call_id: str | None = None,
) -> str | Command:
    """Add a file from host filesystem to the agent's filesystem.
    
    Args:
        file_path: Path to the file on the host filesystem (absolute or relative to cwd).
        backend: Backend instance for the agent's filesystem.
        agent_path: Path in the agent's filesystem where the file should be written.
        tool_call_id: Optional tool call ID for creating ToolMessage in Command response.
    
    Returns:
        Success message string, or Command object if backend returns state updates.
        
    Raises:
        FileNotFoundError: If the file doesn't exist on host filesystem.
        ValueError: If the path is not a file.
        IOError: If there's an error reading or writing the file.
    """
    # Resolve host path
    host_path = Path(file_path)
    if not host_path.is_absolute():
        host_path = Path(os.getcwd()) / host_path
    
    if not host_path.exists():
        raise FileNotFoundError(f"Simulation file not found at {host_path}. Please ensure the file exists.")
    
    if not host_path.is_file():
        raise ValueError(f"Path is not a file: {host_path}")
    
    # Read file content and write via backend
    content = host_path.read_text(encoding='utf-8')
    result: WriteResult = backend.write(agent_path, content)
    
    if result.error:
        raise IOError(f"Error writing to agent filesystem: {result.error}")
    
    message = f"Successfully loaded simulation file and added to agent filesystem at {agent_path}"
    
    # If backend returns state update, return a Command
    if result.files_update is not None:
        return Command(
            update={
                "files": result.files_update,
                "messages": [
                    ToolMessage(
                        content=message,
                        tool_call_id=tool_call_id or "",
                    )
                ],
            }
        )
    
    return message


class PathNormalizationMiddleware(AgentMiddleware[Any, Any]):
    """Middleware that normalizes virtual filesystem paths in shell commands.
    
    Converts virtual paths (starting with `/`) to relative paths since the shell
    executes in the workspace directory. This eliminates path confusion when agents
    copy paths from `ls` output to shell commands.
    """
    
    def __init__(self, shell_tool_name: str = "shell"):
        """Initialize the middleware.
        
        Args:
            shell_tool_name: Name of the shell tool to intercept. Defaults to "shell".
        """
        super().__init__()
        self.shell_tool_name = shell_tool_name
        self.tools = []
    
    def _normalize_paths_in_command(self, command: str) -> str:
        """Normalize virtual paths in a shell command string.
        
        Converts paths like `/filename.py` to `filename.py` while preserving:
        - Quoted paths (single and double quotes)
        - System paths (starting with /usr, /bin, /etc, etc.)
        - Already relative paths
        
        Args:
            command: The shell command string to normalize.
            
        Returns:
            Command string with normalized paths.
        """
        if not command:
            return command
        
        # System path prefixes to preserve (don't normalize)
        system_prefixes = ['/usr', '/bin', '/etc', '/var', '/opt', '/sys', '/proc', '/dev', '/tmp']
        
        # Pattern to match virtual paths: / followed by filename/path
        # Exclude system paths and preserve quoted strings
        def replace_path(match: re.Match) -> str:
            full_match = match.group(0)
            path = match.group(1)
            
            # Don't normalize system paths
            if any(path.startswith(prefix) for prefix in system_prefixes):
                return full_match
            
            # Normalize virtual path by removing leading /
            normalized = path.lstrip('/')
            return full_match.replace(path, normalized)
        
        # Match paths starting with / that are:
        # - Not in quotes (we'll handle quotes separately)
        # - Not system paths
        # Pattern: /filename or /path/to/file (word boundary before /)
        pattern = r'(?<![=\w])(/[a-zA-Z0-9_][a-zA-Z0-9_./-]*)'
        command = re.sub(pattern, replace_path, command)
        
        # Handle quoted paths: "/filename.py" or '/filename.py'
        def replace_quoted_path(match: re.Match) -> str:
            quote = match.group(1)  # ' or "
            path = match.group(2)
            
            # Don't normalize system paths
            if any(path.startswith(prefix) for prefix in system_prefixes):
                return match.group(0)
            
            # Normalize virtual path
            normalized = path.lstrip('/')
            return f'{quote}{normalized}{quote}'
        
        # Match quoted paths: "/path" or '/path'
        quoted_pattern = r'(["\'])(/[a-zA-Z0-9_][a-zA-Z0-9_./-]*)\1'
        command = re.sub(quoted_pattern, replace_quoted_path, command)
        
        return command
    
    def wrap_tool_call(
        self,
        request: Any,  # ToolCallRequest
        handler: Callable[[Any], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept shell tool calls and normalize paths in commands.
        
        Args:
            request: Tool call request with tool_call dict, tool, state, and runtime.
            handler: Callable to execute the tool.
            
        Returns:
            ToolMessage or Command from the handler.
        """
        # Check if this is a shell tool call
        if hasattr(request, 'tool') and hasattr(request.tool, 'name'):
            if request.tool.name == self.shell_tool_name:
                # Normalize paths in the command argument
                if hasattr(request, 'tool_call') and isinstance(request.tool_call, dict):
                    args = request.tool_call.get("args", {})
                    if isinstance(args, dict) and "command" in args:
                        command = args.get("command")
                        if isinstance(command, str):
                            args["command"] = self._normalize_paths_in_command(command)
        
        # Execute the tool with normalized command
        return handler(request)

