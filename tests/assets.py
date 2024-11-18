import dataclasses
import hashlib
import pathlib
import typing

import faery

dirname = (
    pathlib.Path(__file__).resolve().parent
)  # faery.dirname does not work with pytest

Format = typing.Literal[
    "aedat",
    "csv",
    "dat2",
    "es-atis",
    "es-color",
    "es-dvs",
    "es-generic",
    "evt2",
    "evt3",
]

DECODE_DVS_FORMATS: list[Format] = [
    "aedat",
    "csv",
    "dat2",
    "es-atis",
    "es-dvs",
    "evt2",
    "evt3",
]

ENCODE_DVS_FORMATS: list[Format] = [
    "aedat",
    "csv",
    "dat2",
    "es-dvs",
    "evt2",
    "evt3",
]


def format_to_extension(format: Format) -> str:
    if format == "aedat":
        return "aedat"
    if format == "csv":
        return "csv"
    if format == "dat2":
        return "dat"
    if format == "es-atis":
        return "es"
    if format == "es-color":
        return "es"
    if format == "es-dvs":
        return "es"
    if format == "es-generic":
        return "es"
    if format == "evt2":
        return "raw"
    if format == "evt3":
        return "raw"
    raise Exception(f'unsupported format "{format}"')


@dataclasses.dataclass
class File:
    path: pathlib.Path
    format: Format
    field_to_digest: dict[str, str]
    dimensions: typing.Optional[tuple[int, int]]
    time_range: tuple[str, str]
    header_lines: typing.Optional[list[str]]
    tracks: typing.Optional[list[faery.aedat.Track]]
    content_lines: typing.Optional[list[bytes]]
    t0: typing.Optional[str]

    def field_to_hasher(self, fields: typing.Optional[list[str]] = None):
        if fields is None:
            fields = list(self.field_to_digest.keys())
        return {field: hashlib.sha3_224() for field in fields}

    def clone_with(
        self,
        path: pathlib.Path,
        format: Format,
        field_to_digest: dict[str, str],
        tracks: typing.Optional[list[faery.aedat.Track]],
        header_lines: typing.Optional[list[str]],
        content_lines: typing.Optional[list[bytes]],
        t0: typing.Optional[str],
    ) -> "File":
        return File(
            path=path,
            format=format,
            field_to_digest=field_to_digest,
            dimensions=self.dimensions,
            time_range=self.time_range,
            header_lines=header_lines,
            tracks=tracks,
            content_lines=content_lines,
            t0=t0,
        )


files: list[File] = [
    File(
        path=dirname / "data" / "atis.es",
        format="es-atis",
        field_to_digest={
            "atis_t": "74d540ff3fe7dc4b88c6580059feed586418095a8c0deb799927593c",
            "atis_x": "eb78132b1bee84c75243c37e1da72fe3e75b950b61b4b63ea677b153",
            "atis_y": "9e7b7dd6813151f1ed88e9d063e7bacb7209de3681695aece73b60e9",
            "atis_exposure": "40c5268b989d0f2d1b7bbf46b61075df0d7b902bc072abaa3d129840",
            "atis_polarity": "4907bd15c5aff7f3a54b9a82e45503e4bb77078cffe5b05a57542fdc",
            "t": "6f3f9af2e99d83707fd74fef3486abdd9f2f81680d3c4fcc306b1965",
            "x": "976eab4dc499e350481c3f568f71b2713e3ffa944bc1866f31448460",
            "y": "6ccc2ff279b4fb7f87b37d128d84da6771df6b40b624bf3cd4e6622a",
            "on": "6f3c58f949e7c11c55707d11f739d98669882d551403c79482ab81f9",
            "atis_y_original": "a6c8b1035f6c6be4919f95f97256593559ce23d6be9093017d196431",
        },
        dimensions=(320, 240),
        time_range=("00:00:00.000000", "00:00:00.999001"),
        header_lines=None,
        tracks=None,
        content_lines=None,
        t0="00:00:00.000000",
    ),
    File(
        path=dirname / "data" / "color.es",
        format="es-color",
        field_to_digest={
            "t": "6f3f9af2e99d83707fd74fef3486abdd9f2f81680d3c4fcc306b1965",
            "x": "976eab4dc499e350481c3f568f71b2713e3ffa944bc1866f31448460",
            "y": "6ccc2ff279b4fb7f87b37d128d84da6771df6b40b624bf3cd4e6622a",
            "r": "42a8fc278c02481a2dbb625643266dd4f8b5bfd90bb6b059014eb655",
            "g": "23bf9139d88ee294b760ca4c8b496c22201f7dba178c53e2c1ac5f97",
            "b": "960497899f172264075e4b77f6b704f9179443776de594aa4e31bd63",
            "y_original": "6aa82f1dfddd6948a7359816ad62f7fb2c59d00f803e4128868db87b",
        },
        dimensions=(320, 240),
        time_range=("00:00:00.000000", "00:00:00.999001"),
        header_lines=None,
        tracks=None,
        content_lines=None,
        t0="00:00:00.000000",
    ),
    File(
        path=dirname / "data" / "davis346.aedat4",
        format="aedat",
        field_to_digest={
            "t": "f1e093cad5afb6ecb971dfa2ef7646ab4ae0f467f73a48804e40bb68",
            "x": "1d8ea97b0febadfde24dd0b9e608682fc6934fc88656823b15f0e7a7",
            "y": "18f89da35f8f10b24c3407b03aa7f82bdd7c8e6ab5369e2c30f8bad0",
            "on": "6f99cf01187da8a05e1a032f3782de51b87e51bdc31356669bdd7cb9",
            "frame": "6dbd0c0ea251788515bce54edf50b9f29d1995a0330a8b623504379b",
            "imus": "9dffb33769bdb00c67404c3a15479bbd7e204cdc7725976c2ec563ef",
            "triggers": "8479de279528d9d1a04b987ed95d54e2c641124cda618d7072ebc3b7",
        },
        dimensions=(346, 260),
        time_range=("441434:12:27.368868", "441434:12:29.728814"),
        header_lines=[
            '<dv version="2.0">',
            '    <node name="outInfo" path="/mainloop/Recorder/outInfo/">',
            '        <node name="0" path="/mainloop/Recorder/outInfo/0/">',
            '            <attr key="compression" type="string">LZ4</attr>',
            '            <attr key="originalModuleName" type="string">capture</attr>',
            '            <attr key="originalOutputName" type="string">events</attr>',
            '            <attr key="typeDescription" type="string">Array of events (polarity ON/OFF).</attr>',
            '            <attr key="typeIdentifier" type="string">EVTS</attr>',
            '            <node name="info" path="/mainloop/Recorder/outInfo/0/info/">',
            '                <attr key="sizeX" type="int">346</attr>',
            '                <attr key="sizeY" type="int">260</attr>',
            '                <attr key="source" type="string">DAVIS346_00000002</attr>',
            '                <attr key="tsOffset" type="long">1589163005536052</attr>',
            "            </node>",
            "        </node>",
            '        <node name="1" path="/mainloop/Recorder/outInfo/1/">',
            '            <attr key="compression" type="string">LZ4</attr>',
            '            <attr key="originalModuleName" type="string">capture</attr>',
            '            <attr key="originalOutputName" type="string">frames</attr>',
            '            <attr key="typeDescription" type="string">Standard frame (8-bit image).</attr>',
            '            <attr key="typeIdentifier" type="string">FRME</attr>',
            '            <node name="info" path="/mainloop/Recorder/outInfo/1/info/">',
            '                <attr key="sizeX" type="int">346</attr>',
            '                <attr key="sizeY" type="int">260</attr>',
            '                <attr key="source" type="string">DAVIS346_00000002</attr>',
            '                <attr key="tsOffset" type="long">1589163005536052</attr>',
            "            </node>",
            "        </node>",
            '        <node name="2" path="/mainloop/Recorder/outInfo/2/">',
            '            <attr key="compression" type="string">LZ4</attr>',
            '            <attr key="originalModuleName" type="string">capture</attr>',
            '            <attr key="originalOutputName" type="string">imu</attr>',
            '            <attr key="typeDescription" type="string">Inertial Measurement Unit data samples.</attr>',
            '            <attr key="typeIdentifier" type="string">IMUS</attr>',
            '            <node name="info" path="/mainloop/Recorder/outInfo/2/info/">',
            '                <attr key="source" type="string">DAVIS346_00000002</attr>',
            '                <attr key="tsOffset" type="long">1589163005536052</attr>',
            "            </node>",
            "        </node>",
            '        <node name="3" path="/mainloop/Recorder/outInfo/3/">',
            '            <attr key="compression" type="string">LZ4</attr>',
            '            <attr key="originalModuleName" type="string">capture</attr>',
            '            <attr key="originalOutputName" type="string">triggers</attr>',
            '            <attr key="typeDescription" type="string">External triggers and special signals.</attr>',
            '            <attr key="typeIdentifier" type="string">TRIG</attr>',
            '            <node name="info" path="/mainloop/Recorder/outInfo/3/info/">',
            '                <attr key="source" type="string">DAVIS346_00000002</attr>',
            '                <attr key="tsOffset" type="long">1589163005536052</attr>',
            "            </node>",
            "        </node>",
            "    </node>",
            "</dv>",
            "",
        ],
        tracks=[
            faery.aedat.Track(id=0, data_type="events", dimensions=(346, 260)),
            faery.aedat.Track(id=1, data_type="frame", dimensions=(346, 260)),
            faery.aedat.Track(id=2, data_type="imus", dimensions=None),
            faery.aedat.Track(id=3, data_type="triggers", dimensions=None),
        ],
        content_lines=None,
        t0=None,
    ),
    File(
        path=dirname / "data" / "dvs.csv",
        format="csv",
        field_to_digest={
            "t": "6f3f9af2e99d83707fd74fef3486abdd9f2f81680d3c4fcc306b1965",
            "x": "976eab4dc499e350481c3f568f71b2713e3ffa944bc1866f31448460",
            "y": "6ccc2ff279b4fb7f87b37d128d84da6771df6b40b624bf3cd4e6622a",
            "on": "6f3c58f949e7c11c55707d11f739d98669882d551403c79482ab81f9",
        },
        dimensions=(320, 240),
        time_range=("00:00:00.000000", "00:00:00.999001"),
        header_lines=None,
        tracks=None,
        content_lines=None,
        t0="00:00:00.000000",
    ),
    File(
        path=dirname / "data" / "dvs.es",
        format="es-dvs",
        field_to_digest={
            "t": "6f3f9af2e99d83707fd74fef3486abdd9f2f81680d3c4fcc306b1965",
            "x": "976eab4dc499e350481c3f568f71b2713e3ffa944bc1866f31448460",
            "y": "6ccc2ff279b4fb7f87b37d128d84da6771df6b40b624bf3cd4e6622a",
            "on": "6f3c58f949e7c11c55707d11f739d98669882d551403c79482ab81f9",
            "y_original": "6aa82f1dfddd6948a7359816ad62f7fb2c59d00f803e4128868db87b",
        },
        dimensions=(320, 240),
        time_range=("00:00:00.000000", "00:00:00.999001"),
        header_lines=None,
        tracks=None,
        content_lines=None,
        t0="00:00:00.000000",
    ),
    File(
        path=dirname / "data" / "evt2.raw",
        format="evt2",
        field_to_digest={
            "t": "9855ba39c1baea316ae623b72466e420df60a7510248391856eb4eaf",
            "x": "93fc6cff19b483fd48ca53345c6c1cf05123681d7b7272a1717e9303",
            "y": "07cfaef58e5fa71ce9910ed5b88fb7e9fcf71e4a8654cc2c4666f1ae",
            "on": "d2b5e9c4047f4c431937b24cd46c8b504e055372b0ded1df77f0ffa1",
        },
        dimensions=(640, 480),
        time_range=("00:15:13.716224", "00:15:13.812096"),
        header_lines=None,
        tracks=None,
        content_lines=None,
        t0=None,
    ),
    File(
        path=dirname / "data" / "evt3.raw",
        format="evt3",
        field_to_digest={
            "t": "5844f7335f33b60910c5956fac9a6b169414a54f2da8ca2c6d17012d",
            "x": "436762030fbc50f12caaf9fff1fa3a55d3aaff574e8d84159f9eabe6",
            "y": "e7793490007f5b8ad1aa61f3c5adb6af5471af1c40164cd44b17f961",
            "on": "7829ba15f633174ffae397128ce7b2c97bb39bcb7f0042060007c65a",
        },
        dimensions=(1280, 720),
        time_range=("00:00:11.200224", "00:00:21.968222"),
        header_lines=None,
        tracks=None,
        content_lines=None,
        t0=None,
    ),
    File(
        path=dirname / "data" / "gen4.dat",
        format="dat2",
        field_to_digest={
            "t": "78517458e03478cbd6830659dcb09393ba8e013793f2177f18840fc6",
            "x": "436762030fbc50f12caaf9fff1fa3a55d3aaff574e8d84159f9eabe6",
            "y": "e7793490007f5b8ad1aa61f3c5adb6af5471af1c40164cd44b17f961",
            "on": "7829ba15f633174ffae397128ce7b2c97bb39bcb7f0042060007c65a",
        },
        dimensions=(1280, 720),
        time_range=("00:00:00.005856", "00:00:10.773854"),
        header_lines=None,
        tracks=None,
        content_lines=None,
        t0=None,
    ),
    File(
        path=dirname / "data" / "generic.es",
        format="es-generic",
        field_to_digest={
            "t": "4be986c09dccc23887a40a261e8b95ac8c5ab8d0812efd0f78065d78"
        },
        dimensions=None,
        time_range=("00:00:00.000000", "00:00:01.207923"),
        header_lines=None,
        tracks=None,
        content_lines=[
            b"Lorem",
            b"ipsum",
            b"dolor",
            b"sit",
            b"amet,",
            b"consectetur",
            b"adipiscing",
            b"elit,",
            b"sed",
            b"do",
            b"eiusmod",
            b"tempor",
            b"incididunt",
            b"ut",
            b"labore",
            b"et",
            b"dolore",
            b"magna",
            b"aliqua.",
            b"Ut",
            b"enim",
            b"ad",
            b"minim",
            b"veniam,",
            b"quis",
            b"nostrud",
            b"exercitation",
            b"ullamco",
            b"laboris",
            b"nisi",
            b"ut",
            b"aliquip",
            b"ex",
            b"ea",
            b"commodo",
            b"consequat.",
            b"Duis",
            b"aute",
            b"irure",
            b"dolor",
            b"in",
            b"reprehenderit",
            b"in",
            b"voluptate",
            b"velit",
            b"esse",
            b"cillum",
            b"dolore",
            b"eu",
            b"fugiat",
            b"nulla",
            b"pariatur.",
            b"Excepteur",
            b"sint",
            b"occaecat",
            b"cupidatat",
            b"non",
            b"proident,",
            b"sunt",
            b"in",
            b"culpa",
            b"qui",
            b"officia",
            b"deserunt",
            b"mollit",
            b"anim",
            b"id",
            b"est",
            b"laborum.",
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        ],
        t0="00:00:00.000000",
    ),
]
