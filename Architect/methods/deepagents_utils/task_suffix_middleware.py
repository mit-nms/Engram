from typing import Any, Callable
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import AgentMiddleware


class TaskSuffixMiddleware(AgentMiddleware[Any, Any]):
    
    def __init__(self, suffix: str, task_tool_name: str = "task"):
        super().__init__()
        self.suffix = suffix
        self.task_tool_name = task_tool_name
        self.tools = []
    
    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if hasattr(request, 'tool') and hasattr(request.tool, 'name'):
            if request.tool.name == self.task_tool_name:
                if hasattr(request, 'tool_call') and isinstance(request.tool_call, dict):
                    args = request.tool_call.get("args", {})
                    if isinstance(args, dict) and "description" in args:
                        description = args.get("description")
                        if isinstance(description, str):
                            args["description"] = f"{description}\n\n{self.suffix}"
        
        return handler(request)

