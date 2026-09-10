from collections.abc import Callable
from dataclasses import dataclass, field

url = "http://100.107.139.24:1234/v1/chat/completions"
key = "sk-lm-gQhVZVvU:N4TiDU8wD38PPbK99thz"


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Observation:
    tool_name: str
    result: str


@dataclass
class Step:
    action: ToolCall
    observation: Observation


@dataclass
class AgentState:
    task: str
    history: list[Step] = field(default_factory=list)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., object]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: dict) -> str:
        return str(self.handler(**arguments))


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        return self._tools[name].execute(arguments)
