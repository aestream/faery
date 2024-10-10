import enum
import pathlib
import typing


class FileType(enum.Enum):
    AEDAT = 0
    CSV = 1
    DAT = 2
    ES = 3
    EVT = 4

    @staticmethod
    def from_string(string: str):
        string = string.lower()
        if string == "aedat":
            return FileType.AEDAT
        if string == "csv":
            return FileType.CSV
        if string == "dat":
            return FileType.DAT
        if string == "es":
            return FileType.ES
        if string == "evt":
            return FileType.EVT
        raise Exception(f'unknown file format "{string}"')

    def magic(self) -> typing.Optional[bytes]:
        if self == FileType.AEDAT:
            return b"#!AER-DAT4.0\r\n"
        if self == FileType.CSV:
            return None
        if self == FileType.DAT:
            return None
        if self == FileType.ES:
            return b"Event Stream"
        if self == FileType.EVT:
            return None
        raise Exception(f"magic is not implemented for {self}")

    def extensions(self) -> list[str]:
        if self == FileType.AEDAT:
            return [".aedat", ".aedat4"]
        if self == FileType.CSV:
            return [".csv"]
        if self == FileType.DAT:
            return [".dat"]
        if self == FileType.ES:
            return [".es"]
        if self == FileType.EVT:
            return [".evt", ".raw"]
        raise Exception(f"extensions is not implemented for {self}")

    @staticmethod
    def guess(path: pathlib.Path) -> "FileType":
        longest_magic = max(
            0 if magic is None else len(magic)
            for magic in (file_type.magic() for file_type in FileType)
        )
        try:
            with open(path, "rb") as file:
                magic = file.read(longest_magic)
            for file_type in FileType:
                if file_type.magic() == magic:
                    return file_type
        except FileNotFoundError:
            pass
        extension = path.suffix
        for file_type in FileType:
            if any(
                extension == type_extension for type_extension in file_type.extensions()
            ):
                return file_type
        raise Exception(f"unsupported file {path}")
