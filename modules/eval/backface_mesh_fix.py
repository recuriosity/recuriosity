from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

_GLB_MAGIC = b"glTF"
_GLB_JSON_CHUNK_TYPE = 0x4E4F534A  # b"JSON"
_GLB_BIN_CHUNK_TYPE = 0x004E4942  # b"BIN\0"
_GLTF_TRIANGLES_MODE = 4
_CACHE_SCHEMA_VERSION = "v4_backface_black_mode"

_INDEX_COMPONENT_TO_DTYPE = {
    5121: np.dtype("<u1"),  # UNSIGNED_BYTE
    5123: np.dtype("<u2"),  # UNSIGNED_SHORT
    5125: np.dtype("<u4"),  # UNSIGNED_INT
}
_INDEX_COMPONENT_MAX = {
    5121: 255,
    5123: 65535,
    5125: 4294967295,
}


def _source_fingerprint(glb_path: Path) -> str:
    stat = glb_path.stat()
    token = (
        f"{glb_path.resolve()}::"
        f"{int(stat.st_size)}::"
        f"{int(stat.st_mtime_ns)}"
    )
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]


def _build_output_path(
    glb_path: Path,
    cache_dir: Path,
    *,
    force_white: bool,
    backface_black: bool,
) -> Path:
    scene_dir = glb_path.parent.name or "scene"
    stem = glb_path.stem or "stage"
    fp = _source_fingerprint(glb_path)
    white_mode = "white1" if force_white else "white0"
    black_mode = "backblack1" if backface_black else "backblack0"
    return (
        cache_dir
        / scene_dir
        / f"{stem}.{fp}.{white_mode}.{black_mode}.{_CACHE_SCHEMA_VERSION}.hole_fix.glb"
    )


def _parse_glb(glb_bytes: bytes) -> tuple[int, list[tuple[int, bytes]]]:
    if len(glb_bytes) < 12:
        raise ValueError("Invalid GLB: header too short")

    magic, version, declared_length = struct.unpack_from("<4sII", glb_bytes, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("Invalid GLB: bad magic")
    if declared_length != len(glb_bytes):
        raise ValueError(
            f"Invalid GLB: declared length {declared_length} != actual {len(glb_bytes)}"
        )

    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset + 8 <= len(glb_bytes):
        chunk_length, chunk_type = struct.unpack_from("<II", glb_bytes, offset)
        offset += 8
        chunk_end = offset + int(chunk_length)
        if chunk_end > len(glb_bytes):
            raise ValueError("Invalid GLB: chunk extends beyond file length")
        chunks.append((int(chunk_type), glb_bytes[offset:chunk_end]))
        offset = chunk_end

    if offset != len(glb_bytes):
        raise ValueError("Invalid GLB: trailing bytes after final chunk")
    if not chunks:
        raise ValueError("Invalid GLB: no chunks")
    return int(version), chunks


def _decode_json_chunk(chunk_data: bytes) -> dict[str, Any]:
    # JSON chunk is padded with spaces to 4-byte alignment.
    raw = chunk_data.rstrip(b" \t\r\n\x00")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid GLB JSON chunk: top-level payload must be an object")
    return payload


def _encode_json_chunk(payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padding = (-len(raw)) % 4
    return raw + (b" " * padding)


def _encode_glb(version: int, chunks: list[tuple[int, bytes]]) -> bytes:
    body = bytearray()
    for chunk_type, chunk_data in chunks:
        body.extend(struct.pack("<II", int(len(chunk_data)), int(chunk_type)))
        body.extend(chunk_data)
    total_length = 12 + len(body)
    header = struct.pack("<4sII", _GLB_MAGIC, int(version), int(total_length))
    return bytes(header + body)


def _append_aligned(blob: bytearray, data: bytes, *, alignment: int = 4) -> tuple[int, int]:
    if alignment > 1:
        pad = (-len(blob)) % int(alignment)
        if pad:
            blob.extend(b"\x00" * pad)
    offset = len(blob)
    blob.extend(data)
    return int(offset), int(len(data))


def _ensure_primary_buffer(payload: dict[str, Any], *, byte_length: int) -> None:
    buffers = payload.get("buffers")
    if not isinstance(buffers, list) or not buffers:
        payload["buffers"] = [{"byteLength": int(byte_length)}]
        return
    if not isinstance(buffers[0], dict):
        buffers[0] = {}
    buffers[0]["byteLength"] = int(byte_length)


def _get_chunk_index(chunks: list[tuple[int, bytes]], chunk_type: int) -> int | None:
    for idx, (ctype, _data) in enumerate(chunks):
        if int(ctype) == int(chunk_type):
            return int(idx)
    return None


def _get_json_payload(chunks: list[tuple[int, bytes]]) -> tuple[int, dict[str, Any]]:
    json_idx = _get_chunk_index(chunks, _GLB_JSON_CHUNK_TYPE)
    if json_idx is None:
        raise ValueError("Invalid GLB: missing JSON chunk")
    return json_idx, _decode_json_chunk(chunks[json_idx][1])


def _get_or_create_bin_blob(chunks: list[tuple[int, bytes]]) -> tuple[int, bytearray]:
    bin_idx = _get_chunk_index(chunks, _GLB_BIN_CHUNK_TYPE)
    if bin_idx is None:
        chunks.append((_GLB_BIN_CHUNK_TYPE, b""))
        return len(chunks) - 1, bytearray()
    return int(bin_idx), bytearray(chunks[bin_idx][1])


def _choose_index_component_type(max_index: int, preferred: int | None) -> int:
    if preferred in _INDEX_COMPONENT_TO_DTYPE and max_index <= _INDEX_COMPONENT_MAX[int(preferred)]:
        return int(preferred)
    if max_index <= _INDEX_COMPONENT_MAX[5121]:
        return 5121
    if max_index <= _INDEX_COMPONENT_MAX[5123]:
        return 5123
    return 5125


def _read_accessor_indices(
    payload: dict[str, Any],
    primitive: dict[str, Any],
    bin_blob: bytes,
) -> tuple[np.ndarray | None, int | None]:
    accessors = payload.get("accessors")
    buffer_views = payload.get("bufferViews")
    if not isinstance(accessors, list):
        return None, None
    if not isinstance(buffer_views, list):
        return None, None

    if "indices" in primitive:
        try:
            idx_accessor_idx = int(primitive["indices"])
        except Exception:
            return None, None
        if idx_accessor_idx < 0 or idx_accessor_idx >= len(accessors):
            return None, None
        accessor = accessors[idx_accessor_idx]
        if not isinstance(accessor, dict):
            return None, None
        component_type = int(accessor.get("componentType", -1))
        dtype = _INDEX_COMPONENT_TO_DTYPE.get(component_type)
        if dtype is None:
            return None, None
        try:
            count = int(accessor.get("count", 0))
            bview_idx = int(accessor["bufferView"])
        except Exception:
            return None, None
        if count <= 0 or (count % 3) != 0:
            return None, None
        if bview_idx < 0 or bview_idx >= len(buffer_views):
            return None, None
        bview = buffer_views[bview_idx]
        if not isinstance(bview, dict):
            return None, None
        if int(bview.get("buffer", 0)) != 0:
            return None, None
        byte_stride = bview.get("byteStride", None)
        if byte_stride not in (None, dtype.itemsize):
            return None, None
        byte_offset = int(bview.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        byte_length = int(count) * int(dtype.itemsize)
        if byte_offset < 0 or (byte_offset + byte_length) > len(bin_blob):
            return None, None
        indices = np.frombuffer(
            bin_blob,
            dtype=dtype,
            count=int(count),
            offset=int(byte_offset),
        ).astype(np.uint32, copy=True)
        return indices, int(component_type)

    attributes = primitive.get("attributes")
    if not isinstance(attributes, dict):
        return None, None
    if "POSITION" not in attributes:
        return None, None
    try:
        pos_accessor_idx = int(attributes["POSITION"])
    except Exception:
        return None, None
    if pos_accessor_idx < 0 or pos_accessor_idx >= len(accessors):
        return None, None
    pos_accessor = accessors[pos_accessor_idx]
    if not isinstance(pos_accessor, dict):
        return None, None
    try:
        vertex_count = int(pos_accessor.get("count", 0))
    except Exception:
        return None, None
    if vertex_count <= 0 or (vertex_count % 3) != 0:
        return None, None
    return np.arange(vertex_count, dtype=np.uint32), None


def _append_index_accessor(
    payload: dict[str, Any],
    bin_blob: bytearray,
    indices: np.ndarray,
    *,
    preferred_component_type: int | None,
) -> int:
    if not isinstance(payload.get("bufferViews"), list):
        payload["bufferViews"] = []
    if not isinstance(payload.get("accessors"), list):
        payload["accessors"] = []
    buffer_views: list[dict[str, Any]] = payload["bufferViews"]
    accessors: list[dict[str, Any]] = payload["accessors"]

    max_index = int(indices.max(initial=0))
    min_index = int(indices.min(initial=0))
    component_type = _choose_index_component_type(max_index, preferred_component_type)
    dtype = _INDEX_COMPONENT_TO_DTYPE[component_type]
    raw = indices.astype(dtype, copy=False).tobytes(order="C")
    byte_offset, byte_length = _append_aligned(bin_blob, raw, alignment=4)

    bview_idx = len(buffer_views)
    buffer_views.append(
        {
            "buffer": 0,
            "byteOffset": int(byte_offset),
            "byteLength": int(byte_length),
            "target": 34963,  # ELEMENT_ARRAY_BUFFER
        }
    )
    accessor_idx = len(accessors)
    accessors.append(
        {
            "bufferView": int(bview_idx),
            "byteOffset": 0,
            "componentType": int(component_type),
            "count": int(indices.size),
            "type": "SCALAR",
            "min": [int(min_index)],
            "max": [int(max_index)],
        }
    )
    return int(accessor_idx)


def _ensure_backface_material(payload: dict[str, Any], *, use_white: bool) -> int:
    materials = payload.get("materials")
    if not isinstance(materials, list):
        materials = []
        payload["materials"] = materials

    color = [1.0, 1.0, 1.0, 1.0] if use_white else [0.0, 0.0, 0.0, 1.0]
    name = (
        "__cc_eval_hole_fix_backface_white"
        if use_white
        else "__cc_eval_hole_fix_backface_black"
    )
    material_idx = len(materials)
    materials.append(
        {
            "name": name,
            "doubleSided": False,
            "pbrMetallicRoughness": {
                "baseColorFactor": color,
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        }
    )
    return int(material_idx)


def _patch_payload_double_sided(payload: dict[str, Any]) -> dict[str, int]:
    materials = payload.get("materials")
    if not isinstance(materials, list):
        payload["materials"] = []
        materials = payload["materials"]

    total_materials = 0
    updated_materials = 0
    for material in materials:
        if not isinstance(material, dict):
            continue
        total_materials += 1
        if material.get("doubleSided") is not True:
            material["doubleSided"] = True
            updated_materials += 1

    return {
        "mode": "material_double_sided",
        "materials_total": int(total_materials),
        "materials_updated": int(updated_materials),
        "backface_primitives_added": 0,
        "primitives_skipped": 0,
    }


def _patch_payload_backface_black(
    payload: dict[str, Any],
    bin_blob: bytearray,
    *,
    use_white: bool,
) -> dict[str, int]:
    meshes = payload.get("meshes")
    mode_name = "backface_white_primitives" if use_white else "backface_black_primitives"
    if not isinstance(meshes, list):
        return {
            "mode": mode_name,
            "materials_total": int(len(payload.get("materials", [])))
            if isinstance(payload.get("materials"), list)
            else 0,
            "materials_updated": 0,
            "backface_primitives_added": 0,
            "primitives_skipped": 0,
        }

    solid_material_idx = _ensure_backface_material(payload, use_white=use_white)
    primitives_added = 0
    primitives_skipped = 0

    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list):
            continue
        original_primitives = list(primitives)
        to_add: list[dict[str, Any]] = []
        for primitive in original_primitives:
            if not isinstance(primitive, dict):
                continue
            mode = int(primitive.get("mode", _GLTF_TRIANGLES_MODE))
            if mode != _GLTF_TRIANGLES_MODE:
                primitives_skipped += 1
                continue

            indices, preferred_comp = _read_accessor_indices(payload, primitive, bytes(bin_blob))
            if indices is None or indices.size == 0 or (indices.size % 3) != 0:
                primitives_skipped += 1
                continue

            reversed_indices = indices.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
            new_accessor_idx = _append_index_accessor(
                payload,
                bin_blob,
                reversed_indices,
                preferred_component_type=preferred_comp,
            )
            new_primitive = dict(primitive)
            new_primitive["indices"] = int(new_accessor_idx)
            new_primitive["material"] = int(solid_material_idx)
            new_primitive["mode"] = int(_GLTF_TRIANGLES_MODE)
            to_add.append(new_primitive)
            primitives_added += 1

        primitives.extend(to_add)

    materials_total = (
        int(len(payload["materials"])) if isinstance(payload.get("materials"), list) else 0
    )
    return {
        "mode": mode_name,
        "materials_total": int(materials_total),
        "materials_updated": 1,  # one solid-color backface material added
        "backface_primitives_added": int(primitives_added),
        "primitives_skipped": int(primitives_skipped),
    }


def _rewrite_glb_with_hole_fix(
    *,
    source_glb: Path,
    output_glb: Path,
    force_white: bool,
    backface_black: bool,
) -> dict[str, int | str]:
    source_bytes = source_glb.read_bytes()
    version, chunks = _parse_glb(source_bytes)
    json_idx, payload = _get_json_payload(chunks)

    if backface_black or force_white:
        use_white_backfaces = bool(force_white) and not bool(backface_black)
        bin_idx, bin_blob = _get_or_create_bin_blob(chunks)
        stats = _patch_payload_backface_black(
            payload,
            bin_blob,
            use_white=use_white_backfaces,
        )
        if (len(bin_blob) % 4) != 0:
            bin_blob.extend(b"\x00" * ((-len(bin_blob)) % 4))
        chunks[bin_idx] = (_GLB_BIN_CHUNK_TYPE, bytes(bin_blob))
        _ensure_primary_buffer(payload, byte_length=len(bin_blob))
    else:
        stats = _patch_payload_double_sided(payload)

    chunks[json_idx] = (_GLB_JSON_CHUNK_TYPE, _encode_json_chunk(payload))
    output_bytes = _encode_glb(version, chunks)

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    tmp_glb = output_glb.with_name(
        f"{output_glb.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}{output_glb.suffix}"
    )
    try:
        tmp_glb.write_bytes(output_bytes)
        os.replace(tmp_glb, output_glb)
    finally:
        if tmp_glb.exists():
            try:
                tmp_glb.unlink()
            except OSError:
                pass

    return stats


def ensure_double_sided_glb_cached(
    source_glb_path: str,
    *,
    cache_dir: str | None = None,
    force_white: bool = False,
    backface_black: bool = False,
) -> dict[str, Any]:
    source_glb = Path(source_glb_path).expanduser().resolve()
    if not source_glb.is_file():
        raise FileNotFoundError(f"Source GLB not found: {source_glb}")

    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir
        else Path(tempfile.gettempdir()) / "curious_camera_eval_hole_fix"
    )
    output_glb = _build_output_path(
        source_glb,
        cache_root,
        force_white=bool(force_white),
        backface_black=bool(backface_black),
    )
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    if output_glb.exists():
        if backface_black:
            mode_name = "backface_black_primitives"
        elif force_white:
            mode_name = "backface_white_primitives"
        else:
            mode_name = "material_double_sided"
        return {
            "output_path": os.fspath(output_glb),
            "source_path": os.fspath(source_glb),
            "reused_existing": True,
            "mode": mode_name,
            "materials_total": None,
            "materials_updated": None,
            "backface_primitives_added": None,
            "primitives_skipped": None,
        }

    patch_stats = _rewrite_glb_with_hole_fix(
        source_glb=source_glb,
        output_glb=output_glb,
        force_white=bool(force_white),
        backface_black=bool(backface_black),
    )
    return {
        "output_path": os.fspath(output_glb),
        "source_path": os.fspath(source_glb),
        "reused_existing": False,
        **patch_stats,
    }
