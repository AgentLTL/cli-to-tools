"""
cli_to_tools/_effects.py – what a command does to files, beyond its own arguments.

With ``Translator(shell_effects=True)`` each call also says:

- ``redirect_to`` / ``redirect_from``: files a redirection writes (``> f``, ``>> f``,
  ``2> f``, ``&> f``, ``cat <<EOF > f``) or reads (``< f``); ``overwrite_to``: those it
  truncates first (``>``, ``&>``; not ``>>``, nor ``/dev/null`` and other devices);
- path arguments made absolute, resolved from the directory the command line has moved
  to: in ``cd sub && rm a``, ``a`` is ``sub/a``. A ``cd`` lasts until the end of its
  subshell (or ``bash -c``); one in a pipeline changes nothing (each part is a subshell);
- ``unknown_paths``: arguments whose files are only known at run time. ``True`` when any
  path argument may be (``xargs rm``, ``find -exec rm {}``), else the list of those that
  hold one (``rm $UNSET`` gives ``["paths"]``; after ``cd $D`` every relative path is);
- for ``patch`` / ``git apply``, the files the diff modifies, as ``paths``;
- for ``curl -O``, the file it writes, as ``output``.

Which arguments hold paths: those a spec marks ``path: true``, and by default those named
in :data:`PATH_NAMES` (a spec can say ``path: false``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Argument names that hold file paths unless a spec says otherwise.
PATH_NAMES = ("path", "paths", "file", "files", "directory", "sources", "destination",
              "destination_dir", "of", "if", "archive", "input", "patchfile", "patches",
              "script_file", "program_file", "output", "file_path", "notebook_path")
# Added by the shell effects themselves.
REDIRECT_KEYS = ("redirect_to", "overwrite_to", "redirect_from")
UNKNOWN_PATHS = "unknown_paths"
_DEVICES = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty")


@dataclass(frozen=True)
class Paths:
    """Where relative paths are resolved from (``cwd``), and the project root, if any."""

    cwd: str = ""
    root: str = ""


def is_device(target: str) -> bool:
    """Writing to these discards or shows the output; it overwrites no file."""
    return target in _DEVICES or target.startswith("/dev/fd/")


def redirect_files(redirects: Sequence[Dict[str, str]]) -> Tuple[List[str], List[str], List[str]]:
    """``(writes, reads, truncates)`` of a command's redirections."""
    writes: List[str] = []
    reads: List[str] = []
    truncates: List[str] = []
    for r in redirects:
        op, target = r.get("op", ""), r.get("target", "")
        if not target:
            continue
        if op in (">&", "<&") and (target.isdigit() or target == "-"):
            continue                       # 2>&1: copies a descriptor, no file
        if op in (">", ">>", ">|", "&>", "&>>", ">&"):
            writes.append(target)
            if op in (">", ">|", "&>", ">&") and not is_device(target):
                truncates.append(target)
        elif op == "<":
            reads.append(target)
    return writes, reads, truncates


def absolute(value: str, paths: Paths) -> str:
    return os.path.normpath(os.path.join(paths.cwd or os.getcwd(), os.path.expanduser(value)))


def canonical(value: str, paths: Paths) -> str:
    """A path argument made absolute; flags, URLs and unexpanded variables are left alone."""
    home = os.path.expanduser("~")
    for prefix in ("${HOME}", "$HOME"):
        if value == prefix or value.startswith(prefix + "/"):
            value = home + value[len(prefix):]
    if not value or value.startswith(("-", "http://", "https://", "$")) or "://" in value:
        return value
    return absolute(value, paths)


def normalize_paths(args: Dict[str, Any], paths: Paths,
                    keys: Iterable[str] = PATH_NAMES + REDIRECT_KEYS) -> Dict[str, Any]:
    """*args* with the path arguments among *keys* made absolute."""
    out = dict(args)
    for key in keys:
        value = out.get(key)
        if isinstance(value, str):
            out[key] = canonical(value, paths)
        elif isinstance(value, list):
            out[key] = [canonical(v, paths) if isinstance(v, str) else v for v in value]
    return out


def strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in strings(v)]
    return []


def unknown_path_keys(args: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    """The path arguments among *keys* holding a value only known at run time."""
    return [key for key in keys if any(
        "$" in v or "`" in v or "{}" in v or "<(" in v for v in strings(args.get(key)))]


def relative_path_keys(args: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    """The path arguments among *keys* holding a relative path."""
    out = []
    for key in keys:
        if any(v and not v.startswith(("/", "~", "$", "-")) and "://" not in v
               for v in strings(args.get(key))):
            out.append(key)
    return out


def cd_target(name: str, args: Dict[str, Any], here: Optional[str]) -> Optional[str]:
    """The directory ``cd`` / ``pushd`` / ``popd`` moves to; None when only known at run
    time (``cd $D``, ``cd -``, ``popd``, or a relative move from an unknown directory)."""
    if name == "popd":
        return None
    if name == "pushd":
        words = [w for w in args.get("argv") or [] if not w.startswith("-")]
        target = words[0] if words else None
        if target is None or target.startswith("+"):
            return None                     # swaps with the directory stack
    else:
        target = args.get("path")
        if not target:
            return os.path.expanduser("~")
    if target == "-" or args.get(UNKNOWN_PATHS) or "$" in target or "`" in target:
        return None
    target = os.path.expanduser(target)
    if os.path.isabs(target):
        return os.path.normpath(target)
    return None if here is None else os.path.normpath(os.path.join(here, target))


def patch_targets(name: str, args: Dict[str, Any], paths: Paths) -> None:
    """Read the diff a ``patch`` / ``git apply`` will apply and list the files it touches."""
    diffs = [args.get("input"), args.get("patchfile"), *(args.get("patches") or []),
             *(args.get("redirect_from") or [])]
    diffs = [d for d in diffs if d]
    strip = args.get("strip")
    strips = [int(strip)] if str(strip or "").isdigit() else ([1] if name == "git_apply" else [0, 1])
    found: List[str] = []
    readable = bool(diffs)
    for diff in diffs:
        path = os.path.join(paths.cwd or os.getcwd(), os.path.expanduser(diff))
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read(2_000_000)
        except OSError:
            readable = False
            continue
        for file in diff_files(text):
            for n in strips:
                parts = file.split("/")
                if len(parts) > n:
                    found.append("/".join(parts[n:]))
    if found:
        args["paths"] = list(dict.fromkeys(found))
    if not readable:
        args[UNKNOWN_PATHS] = True


def diff_files(text: str) -> List[str]:
    """The files a unified or git diff names."""
    names: List[str] = []
    for line in text.splitlines():
        if line.startswith(("--- ", "+++ ")):
            name = line[4:].split("\t")[0].strip()
        elif line.startswith(("rename to ", "rename from ", "copy to ")):
            name = line.split(" ", 2)[2].strip()
        else:
            continue
        if name and name != "/dev/null":
            names.append(name)
    return list(dict.fromkeys(names))


def curl_output(args: Dict[str, Any]) -> None:
    """``curl -O URL`` writes the URL's basename (in ``--output-dir``, if given)."""
    from urllib.parse import urlparse
    names = [os.path.basename(urlparse(u).path) for u in args.get("urls") or []]
    names = [n for n in names if n]
    if names:
        out_dir = args.get("output_dir") or ""
        args["output"] = [os.path.join(out_dir, n) for n in names]


class Effects:
    """Applies the shell effects to the calls of one command line, in order."""

    def __init__(self, paths: Optional[Paths]) -> None:
        self.paths = paths
        self._dirs: Dict[int, Optional[str]] = {}    # subshell depth -> directory (None: unknown)

    def apply(self, name: str, args: Dict[str, Any], meta: Dict[str, Any],
              path_keys: Sequence[str]) -> Dict[str, Any]:
        writes, reads, truncates = redirect_files(meta.get("redirects") or [])
        if writes:
            args["redirect_to"] = writes
        if truncates:
            args["overwrite_to"] = truncates
        if reads:
            args["redirect_from"] = reads
        keys = tuple(dict.fromkeys([*path_keys, *REDIRECT_KEYS]))
        here: Optional[str] = None
        lost = False
        if self.paths is not None:
            depth = meta.get("depth") or 0
            for d in [d for d in self._dirs if d > depth]:
                del self._dirs[d]                   # left that subshell
            here = self._dirs[max(self._dirs)] if self._dirs else self.paths.cwd
            lost = here is None
            paths = Paths(here or self.paths.cwd, self.paths.root)
            relative = relative_path_keys(args, keys) if lost else []
            if name in ("patch", "git_apply"):
                patch_targets(name, args, paths)
            if name == "curl" and args.get("remote_name") and not args.get("output"):
                curl_output(args)
            args = normalize_paths(args, paths, keys)
        else:
            relative = []
        if meta.get("wrapper") == "xargs":
            args[UNKNOWN_PATHS] = True
        elif not args.get(UNKNOWN_PATHS):
            unknown = list(dict.fromkeys(unknown_path_keys(args, keys) + relative))
            if unknown:
                args[UNKNOWN_PATHS] = unknown
        if self.paths is not None and name in ("cd", "pushd", "popd") \
                and not meta.get("pipeline"):
            self._dirs[meta.get("depth") or 0] = cd_target(name, args, here)
        return args
