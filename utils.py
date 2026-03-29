from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional


SUPPORTED_TABULAR_SUFFIXES = {".xlsx", ".xls", ".csv"}


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "读取表格文件需要安装 pandas；.xlsx 建议 openpyxl；部分 WPS 导出的 .xls 在"
            " xlrd 下会解码失败，可安装 python-calamine 并由本模块优先用 calamine 引擎读取。"
        ) from exc
    return pd


def _load_dataframe(pd: Any, file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, encoding="utf-8-sig")
    if suffix == ".xlsx":
        return pd.read_excel(file_path, engine="openpyxl")
    if suffix == ".xls":
        # WPS 等工具生成的 .xls 常使 xlrd 在解析共享字符串表时触发 UTF-16 解码错误；
        # calamine 引擎对这些文件兼容性更好。未安装 python-calamine 时回退为默认（通常为 xlrd）。
        try:
            return pd.read_excel(file_path, engine="calamine")
        except Exception:
            return pd.read_excel(file_path)
    raise ValueError(f"不支持的文件类型: {file_path}")


def _normalize_max_items(max_items: Optional[int]) -> Optional[int]:
    if max_items is None:
        return None
    if max_items <= 0:
        return 0
    return max_items


def _append_unique_value(
    pd: Any, values: List[str], seen: set[str], raw_value: object, max_items: Optional[int]
) -> bool:
    if pd.isna(raw_value):
        return False

    value = str(raw_value).strip()
    if not value or value in seen:
        return False

    seen.add(value)
    values.append(value)
    return max_items is not None and len(values) >= max_items


def _read_unique_column_values(
    file_path: Path, column_name: str, max_items: Optional[int] = None
) -> List[str]:
    pd = _import_pandas()

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"不是文件: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_TABULAR_SUFFIXES:
        raise ValueError(
            f"不是支持的表格文件（.xlsx / .xls / .csv）: {file_path}"
        )

    max_items = _normalize_max_items(max_items)
    if max_items == 0:
        return []

    df = _load_dataframe(pd, file_path)
    if column_name not in df.columns:
        raise ValueError(
            f"文件 {file_path} 中不存在字段 {column_name!r}，可用字段: {list(df.columns)}"
        )

    values: List[str] = []
    seen: set[str] = set()
    for raw_value in df[column_name].tolist():
        reached_limit = _append_unique_value(pd, values, seen, raw_value, max_items)
        if reached_limit:
            break

    return values


def read_unique_titles_from_excel(
    excel_path: str | Path,
    column_name: str = "caption",
    max_items: Optional[int] = None,
) -> List[str]:
    """
    读取单个表格文件中指定字段的所有值，去重、去空后返回字符串列表。

    Args:
        excel_path: 文件路径，支持 .xlsx、.xls 与 .csv（CSV 默认按 UTF-8，含 BOM 时用 utf-8-sig）。
        column_name: 要读取的字段名，默认值为 ``caption``。
        max_items: 最大返回数量；为 ``None`` 时表示不限制。
    """
    return _read_unique_column_values(
        file_path=Path(excel_path),
        column_name=column_name,
        max_items=max_items,
    )


def read_unique_titles_from_subdirs(
    root_dir: str | Path,
    column_name: str = "视频标题",
    max_items: Optional[int] = None,
) -> List[str]:
    """
    读取目录下所有一级子目录中的表格文件（.xlsx / .xls / .csv），合并指定字段的值后去重、去空。

    Args:
        root_dir: 根目录路径，只扫描其一级子目录中的上述格式文件。
        column_name: 要读取的字段名，默认值为 ``视频标题``。
        max_items: 最大返回数量；为 ``None`` 时表示不限制。
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"目录不存在: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"不是目录: {root_path}")

    max_items = _normalize_max_items(max_items)
    if max_items == 0:
        return []

    data_files: List[Path] = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        for file_path in sorted(child.iterdir()):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in SUPPORTED_TABULAR_SUFFIXES
            ):
                data_files.append(file_path)

    titles: List[str] = []
    seen: set[str] = set()

    for data_file in data_files:
        file_titles = _read_unique_column_values(
            file_path=data_file,
            column_name=column_name,
            max_items=None,
        )
        for title in file_titles:
            if title in seen:
                continue
            seen.add(title)
            titles.append(title)
            if max_items is not None and len(titles) >= max_items:
                return titles

    return titles


def read_unique_titles(
    path: str | Path,
    column_name: Optional[str] = None,
    max_items: Optional[int] = None,
) -> List[str]:
    """
    根据 ``path`` 是文件还是目录，自动选择读取方式，便于外部单一入口调用。

    - **文件**：调用 ``read_unique_titles_from_excel``（支持 .xlsx / .xls / .csv）；未传 ``column_name`` 时默认 ``caption``。
    - **目录**：调用 ``read_unique_titles_from_subdirs``；未传 ``column_name`` 时默认 ``视频标题``。
    """
    target = Path(path)
    if target.is_file():
        col = column_name if column_name is not None else "caption"
        return read_unique_titles_from_excel(
            excel_path=target,
            column_name=col,
            max_items=max_items,
        )
    if target.is_dir():
        col = column_name if column_name is not None else "视频标题"
        return read_unique_titles_from_subdirs(
            root_dir=target,
            column_name=col,
            max_items=max_items,
        )
    raise FileNotFoundError(f"路径不存在或不是文件/目录: {target}")


if __name__ == "__main__":
    titles = read_unique_titles(
        path="data/douyin",
        column_name="视频标题",
        max_items=3000,
    )
    for title in titles:
        print(title)
        