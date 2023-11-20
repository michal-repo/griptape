from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
from attr import define, field, Factory
import logging
from rich import print
from rich.live import Live
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from griptape.utils.stream import Stream
from griptape.events import EventListener, CompletionChunkEvent, StartTaskEvent, BaseEvent

if TYPE_CHECKING:
    from griptape.structures import Structure


@define(frozen=True)
class Chat:
    structure: Structure = field()
    exit_keywords: list[str] = field(default=["exit"], kw_only=True)
    exiting_text: str = field(default="exiting...", kw_only=True)
    processing_text: str = field(default="processing...", kw_only=True)
    intro_text: Optional[str] = field(default=None, kw_only=True)
    prompt_prefix: str = field(default="Q: ", kw_only=True)
    response_prefix: str = field(default="A: ", kw_only=True)
    streaming_output_fn: Callable[[str], None] = field(default=lambda x: print(x, end=""), kw_only=True)
    output_fn: Callable[[str], None] = field(default=Factory(lambda: print), kw_only=True)

    def start(self) -> None:
        if self.intro_text:
            print(Panel(Text(self.intro_text, style="light_steel_blue3"), title="Intro"))
        while True:
            question = Prompt.ask(Text(self.prompt_prefix, style="light_steel_blue3"))

            if question.lower() in self.exit_keywords:
                self.output_fn(self.exiting_text)
                break
            else:
                self.output_fn(self.processing_text)

            if self.structure.prompt_driver.stream:
                global output
                output = ""

                def generate_layout(structure: Structure, output: str) -> Layout:
                    layout = Layout()
                    layouts = []
                    for task in structure.tasks:
                        if task.output:
                            layout_id = f"task_{task.id}"
                            finished_layout = Layout(
                                Panel(Markdown(task.output.to_text(), style="light_steel_blue3"), title=layout_id),
                                name=layout_id,
                            )
                            layouts.append(finished_layout)

                    layouts.append(
                        Layout(Panel(Text(output, style="light_steel_blue3"), title="Output"), name="output")
                    )

                    layout.split_column(*layouts)

                    return layout

                with Live(generate_layout(self.structure, "")) as live:

                    def event_handler(e: BaseEvent):
                        global output, title
                        if isinstance(e, CompletionChunkEvent):
                            output += e.token
                        elif isinstance(e, StartTaskEvent):
                            output = ""
                        live.update(generate_layout(self.structure, output))

                    stream_event_listener = EventListener(
                        lambda e: event_handler(e), event_types=[CompletionChunkEvent, StartTaskEvent]
                    )
                    self.structure.logger_level = logging.ERROR
                    self.structure.add_event_listener(stream_event_listener)
                    self.structure.run(question)
                    self.structure.remove_event_listener(stream_event_listener)
            else:
                self.output_fn(f"{self.response_prefix}{self.structure.run(question).output.to_text()}")
