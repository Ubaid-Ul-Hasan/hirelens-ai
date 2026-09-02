"""
Import shim for pydantic.

In the target environment (see requirements.txt), real `pydantic` is
installed and used directly. This shim exists ONLY so the project's own
logic can be exercised in constrained/offline sandboxes that lack network
access to install pydantic -- it is not a replacement for real validation
and should not be relied upon in production.

Usage elsewhere in the codebase:
    from src._pydantic_compat import BaseModel, Field
"""
try:  # pragma: no cover - trivial branch
    from pydantic import BaseModel, Field  # type: ignore  # noqa: F401

    PYDANTIC_AVAILABLE = True

except ImportError:  # pragma: no cover
    import copy
    from typing import Any, get_type_hints

    PYDANTIC_AVAILABLE = False

    class _FieldInfo:
        def __init__(self, default=None, default_factory=None):
            self.default = default
            self.default_factory = default_factory

    def Field(default: Any = None, default_factory=None, **_kwargs):  # noqa: N802
        return _FieldInfo(default=default, default_factory=default_factory)

    class BaseModel:
        """Minimal stand-in supporting the subset of pydantic v1/v2 API
        this project uses: typed attributes with defaults, .dict()/.model_dump(),
        keyword-arg construction, and nested BaseModel instantiation from
        plain dicts.
        """

        def __init__(self, **data: Any):
            hints = get_type_hints(self.__class__)
            for field_name in hints:
                class_default = getattr(self.__class__, field_name, None)
                if field_name in data:
                    value = data[field_name]
                else:
                    if isinstance(class_default, _FieldInfo):
                        if class_default.default_factory is not None:
                            value = class_default.default_factory()
                        else:
                            value = copy.deepcopy(class_default.default)
                    else:
                        value = copy.deepcopy(class_default)
                setattr(self, field_name, value)

        def dict(self) -> dict:
            out = {}
            for k, v in self.__dict__.items():
                if isinstance(v, BaseModel):
                    out[k] = v.dict()
                elif isinstance(v, list):
                    out[k] = [
                        item.dict() if isinstance(item, BaseModel) else item
                        for item in v
                    ]
                else:
                    out[k] = v
            return out

        model_dump = dict  # pydantic v2 style alias

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self.dict()})"
