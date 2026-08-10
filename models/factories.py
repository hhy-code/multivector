from collections.abc import Callable
from typing import Any
from typing import TypeVar

from utils.utils import load_config

from .base import BaseModel, BaseTokenizer


ModelT = TypeVar("ModelT", bound=type)
TokenizerT = TypeVar("TokenizerT", bound=type)

MODEL_CLASSES: dict[str, type] = {}
TOKENIZER_CLASSES: dict[str, type] = {}


def register_model(name: str) -> Callable[[ModelT], ModelT]:
    def decorator(cls: ModelT) -> ModelT:
        MODEL_CLASSES[name] = cls
        return cls

    return decorator


def register_tokenizer(name: str) -> Callable[[TokenizerT], TokenizerT]:
    def decorator(cls: TokenizerT) -> TokenizerT:
        TOKENIZER_CLASSES[name] = cls
        return cls

    return decorator


def create_model(name: str, config: str | dict[str, Any]) -> BaseModel:
    config = load_config(config)

    if "model" in config:
        config = config["model"]

    model_class = MODEL_CLASSES.get(name)
    if model_class is None:
        raise ValueError(
            f"Model class for '{name}' not found. Available models: {list(MODEL_CLASSES)}"
        )
    return model_class.from_pretrained(config)


def create_tokenizer(name: str, config: str | dict[str, Any]) -> BaseTokenizer:
    config = load_config(config)

    if "tokenizer" in config:
        config = config["tokenizer"]

    tokenizer_class = TOKENIZER_CLASSES.get(name)
    if tokenizer_class is None:
        raise ValueError(
            f"Tokenizer class for '{name}' not found. Available tokenizers: {list(TOKENIZER_CLASSES)}"
        )
    return tokenizer_class.from_config(config)


def create_model_and_tokenizer(
    name: str, config: str | dict[str, Any]
) -> tuple[BaseModel, BaseTokenizer]:
    config = load_config(config)

    if "model" in config:
        model_config = config["model"]
    else:
        model_config = config

    if "tokenizer" in config:
        tokenizer_config = config["tokenizer"]
    else:
        tokenizer_config = config

    return create_model(name, model_config), create_tokenizer(name, tokenizer_config)