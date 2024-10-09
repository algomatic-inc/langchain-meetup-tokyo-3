import argparse

import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from langchain_api.hello_world import hello_world_chain
from langchain_api.nobunaga_chat import nobunaga_chat_chain
from langchain_api.visual_tagging import visual_tagging_chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = FastAPI()

    add_routes(app, hello_world_chain, path="/hello_world")
    add_routes(app, nobunaga_chat_chain, path="/nobunaga_chat", playground_type="chat")
    add_routes(app, visual_tagging_chain, path="/visual_tagging")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
