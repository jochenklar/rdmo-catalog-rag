import argparse
import logging

from dotenv import load_dotenv
from rich.logging import RichHandler

from .store import create_vector_store, query_vector_store

load_dotenv(".env")

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new vector store")
    create_parser.set_defaults(func=create_vector_store)

    query_parser = subparsers.add_parser("query", help="Query the vector store")
    query_parser.add_argument("query", help="Query text")
    query_parser.set_defaults(func=query_vector_store)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
