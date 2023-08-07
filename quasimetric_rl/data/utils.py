from __future__ import annotations
from typing import *
from typing_extensions import Self, get_args, get_origin

import abc
import attrs

import torch
import torch.utils.data


vT = TypeVar('vT', covariant=True)
class NestedMapping(Mapping[str, Union['NestedMapping[vT]', vT]]):
    def __getitem__(self, __key: str) -> Union[Self, vT]:
        return super().__getitem__(__key)


FieldT = TypeVar(
    'FieldT',
    torch.Tensor, NestedMapping[torch.Tensor],
    'TensorCollectionAttrsMixin', NestedMapping['TensorCollectionAttrsMixin'],
)


class TensorCollectionAttrsMixin(abc.ABC):
    # All fields must be one of
    #    torch.Tensor
    #    NestedMapping[torch.Tensor]
    #    TensorCollectionAttrsMixin
    #    NestedMapping[TensorCollectionAttrsMixin]

    @classmethod
    def types_dict(cls):
        fields = attrs.fields_dict(cls)
        return {k: t for k, t in get_type_hints(cls).items() if k in fields}

    @staticmethod
    def is_tensor_type(ty):
        try:
            return issubclass(ty, torch.Tensor)
        except TypeError:
            return False

    @staticmethod
    def is_nested_tensor_mapping_type(ty):
        try:
            orig = get_origin(ty)
            args = get_args(ty)
            return (
                (issubclass(orig, NestedMapping) and issubclass(args)[0], torch.Tensor)
                or
                (issubclass(orig, Mapping) and args[0] == str and issubclass(args, torch.Tensor))
            )
        except TypeError:
            return False

    @staticmethod
    def is_tensor_collection_attrs_type(ty):
        try:
            return issubclass(ty, TensorCollectionAttrsMixin)
        except TypeError:
            return False

    @classmethod
    def cat(cls, collections: List[Self], *, dim=0) -> Self:
        assert all(isinstance(c, cls) for c in collections)

        if len(collections) == 1:  # differ from torch.cat: no copy
            return collections[0]

        types = cls.types_dict()

        def cat_key(k: str):
            ty = types[k]
            field_values = [getattr(c, k) for c in collections]
            if cls.is_tensor_type(ty):
                # torch.Tensor
                return torch.cat(field_values, dim=dim)  # differ from torch.cat: no copy if len == 1
            elif cls.is_nested_tensor_mapping_type(ty):
                # NestedMapping[torch.Tensor]

                def cat_map(maps: List[NestedMapping[torch.Tensor]]) -> NestedMapping[torch.Tensor]:
                    if len(maps) == 0:
                        return {}

                    def get_tensor_flags(map: NestedMapping[torch.Tensor]):
                        return {map_k: isinstance(map_v, torch.Tensor) for map_k, map_v in map.items()}

                    tensor_flags = get_tensor_flags(maps[0])

                    return {
                        map_k: (
                            torch.cat([m[map_k] for m in maps], dim=dim) if is_tensor else cat_map([m[map_k] for m in maps])
                        ) for map_k, is_tensor in tensor_flags.items()
                    }

                return cat_map(field_values)
            elif cls.is_tensor_collection_attrs_type(ty):
                # TensorCollectionAttrsMixin
                return cast(Type[TensorCollectionAttrsMixin], ty).cat(field_values, dim=dim)
            else:
                # NestedMapping[TensorCollectionAttrsMixin]

                coll_ty: Type[TensorCollectionAttrsMixin] = get_args(ty)[0]

                def cat_map(maps: List[NestedMapping[TensorCollectionAttrsMixin]]) -> NestedMapping[TensorCollectionAttrsMixin]:
                    if len(maps) == 0:
                        return {}

                    def get_coll_flags(map: NestedMapping[TensorCollectionAttrsMixin]):
                        return {map_k: isinstance(map_v, TensorCollectionAttrsMixin) for map_k, map_v in map.items()}

                    coll_flags = get_coll_flags(maps[0])

                    return {
                        map_k: (
                            coll_ty.cat([m[map_k] for m in maps], dim=dim) if is_coll else cat_map([m[map_k] for m in maps])
                        ) for map_k, is_coll in coll_flags.items()
                    }

                return cat_map(field_values)

        return cls(**{k: cat_key(k) for k in types.keys()})

    @staticmethod
    def _make_cvt_fn(elem_cvt_fn: Callable[[Union[torch.Tensor, TensorCollectionAttrsMixin]], Union[torch.Tensor, TensorCollectionAttrsMixin]]):
        def cvt_fn(x: FieldT) -> FieldT:
            if isinstance(x, (torch.Tensor, TensorCollectionAttrsMixin)):
                return elem_cvt_fn(x)
            else:
                return {k: cvt_fn(v) for k, v in x.items()}
        return cvt_fn

    def to(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.to(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def flatten(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.flatten(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def unflatten(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.unflatten(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def narrow(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.narrow(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def __getitem__(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.__getitem__(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def pin_memory(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.pin_memory(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )
