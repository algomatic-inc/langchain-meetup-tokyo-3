from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class VisualTaggingInput(BaseModel):
    image_url: str = Field(description="URL of the image to tag")
    language: Literal["english", "japanese"] = Field(
        description="Language of the output"
    )


class VisualTaggingOutput(BaseModel):
    title: str = Field(..., description="Title of the image")
    summary: str = Field(..., description="Summary of the image")
    tags: list[str] = Field(..., description="List of tags for the image")


english_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that tags images. Please provide a title, summary, and tags for the image.",  # noqa: E501
        ),
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "{image_url}",
                    },
                }
            ],
        ),
    ]
)

japanese_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "画像のタイトル、要約、タグを**必ず**日本語でつけてください。",  # noqa: E501
        ),
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "{image_url}",
                    },
                }
            ],
        ),
    ]
)

chat = (
    ChatOpenAI(model="gpt-4o-2024-08-06", top_p=0)
    .with_structured_output(VisualTaggingOutput, method="json_schema", strict=True)
    .bind(seed=0)
)


def route_prompt(input: dict) -> ChatPromptTemplate:
    if input["language"] == "english":
        return english_prompt
    elif input["language"] == "japanese":
        return japanese_prompt
    else:
        raise ValueError(f"Invalid language: {input['language']}")


visual_tagging_chain = (
    RunnableLambda(route_prompt).with_types(input_type=VisualTaggingInput) | chat
)
