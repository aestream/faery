import typing


class Node:
    def __init__(self, type: str, attributes: dict[str, typing.Any], indent: int):
        self.type = type
        self.attributes = attributes
        self.indent = indent
        self.children: list[typing.Union[Node, str]] = []

    def node(self, type: str, attributes: dict[str, typing.Any]) -> "Node":
        child = Node(type=type, attributes=attributes, indent=self.indent + 1)
        self.children.append(child)
        return child

    def text(self, value: str):
        self.children.append(value)

    def to_string(self, indent: str = "    ", line_breaks: bool = True) -> str:
        if line_breaks:
            end = "\n"
        else:
            end = ""
        attributes = " ".join(
            f'{name}="{value}"' for name, value in self.attributes.items()
        )
        if len(self.children) == 0:
            return f"{indent * self.indent}<{self.type} {attributes} />{end}"
        else:
            if line_breaks and any(isinstance(child, Node) for child in self.children):
                after_tag = "\n"
                before_closing_tag = f"{indent * self.indent}"
            else:
                after_tag = ""
                before_closing_tag = ""
            return "".join(
                (
                    f"{indent * self.indent}<{self.type} {attributes}>{after_tag}",
                    *(
                        (
                            child.to_string(indent=indent, line_breaks=line_breaks)
                            if isinstance(child, Node)
                            else child
                        )
                        for child in self.children
                    ),
                    f"{before_closing_tag}</{self.type}>{end}",
                )
            )


class Svg(Node):
    def __init__(self, width: int, height: int):
        super().__init__(
            type="svg",
            attributes={
                "xmlns": "http://www.w3.org/2000/svg",
                "width": width,
                "height": height,
                "viewBox": f"0 0 {width} {height}",
            },
            indent=0,
        )
