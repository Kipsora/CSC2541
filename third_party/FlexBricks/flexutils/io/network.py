__all__ = ['download']

import os
from http import HTTPStatus
from typing import Optional

import requests
import tqdm

from flexutils.cli.colored import CLIColorFormat
from flexutils.io.file_system import *


def download(url: str,
             path: str,
             filename: Optional[str] = None,
             show_progress_bar: bool = True,
             force_redownload: bool = False,
             chunk_size: int = 1024 * 1024,
             **kwargs):
  try:
    with requests.head(url, allow_redirects=True) as response:
      if not response.ok:
        raise RuntimeError(f"HTTP Request error: {response.reason}")
      filename = filename or os.path.basename(response.url) or "index.html"
      filepath = os.path.join(path, filename)
      num_total_bytes = None
      if "Content-Length" in response.headers:
        num_total_bytes = int(response.headers["Content-Length"])

      num_downloaded_bytes = 0
      if os.path.exists(filepath) and not force_redownload:
        num_downloaded_bytes = os.path.getsize(filepath)
        if num_total_bytes is not None:
          if num_total_bytes < num_downloaded_bytes:
            raise RuntimeError(
                f"The number of downloaded bytes is larger than total bytes ("
                f"downloaded: {num_downloaded_bytes}, total: {num_total_bytes})"
            )
    if num_total_bytes is None or num_downloaded_bytes < num_total_bytes:
      headers = {'Range': f'bytes={num_downloaded_bytes}-'}
      with requests.get(url, stream=True, allow_redirects=True,
                        headers=headers) as response:
        if response.status_code != HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE \
                or num_total_bytes is not None:
          if not response.ok:
            raise RuntimeError(f"HTTP Request error: {response.reason}")
          content_length = response.headers.get("Content-Length", None)
          if content_length is not None and content_length.isdigit():
            num_remaining_bytes = int(content_length)
            if num_total_bytes is None:
              num_total_bytes = num_downloaded_bytes + num_remaining_bytes
            if num_remaining_bytes + num_downloaded_bytes != num_total_bytes:
              raise RuntimeError(f"Inconsistent total size ("
                                 f"downloaded: {num_downloaded_bytes} "
                                 f"remaining: {num_remaining_bytes} "
                                 f"total: {num_total_bytes})")
          bar = None
          if show_progress_bar:
            header = CLIColorFormat(bold=True).colored("Downloading")
            colored_path = CLIColorFormat(underline=True, color="green")
            colored_path = colored_path.colored(simplify_path(filepath))
            bar = tqdm.tqdm(total=num_total_bytes,
                            initial=num_downloaded_bytes,
                            unit='B',
                            unit_scale=True,
                            leave=False,
                            miniters=1,
                            desc=f"{header} {colored_path}",
                            dynamic_ncols=True)
          try:
            ensure_directory(path)
            with open(filepath, 'wb' if force_redownload else 'ab') as writer:
              for data in response.iter_content(chunk_size=chunk_size):
                writer.write(data)
                if bar is not None:
                  bar.update(len(data))
          finally:
            if bar is not None:
              bar.close()

    for key in kwargs:
      checksum = get_file_checksum(filepath,
                                   key,
                                   chunk_size,
                                   show_progress_bar=show_progress_bar)
      if checksum != kwargs[key]:
        raise RuntimeError(f"Integrity check failed ({key} checksum is "
                           f"{checksum}, but {kwargs[key]}) is expected")
  except Exception as exception:
    raise IOError(f"Failed to download {url}") from exception

  return filepath
