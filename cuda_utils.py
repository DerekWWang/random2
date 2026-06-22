"""
CUDA kernel utilities for Windows + conda (deepl env).

Usage in any notebook:
    from cuda_utils import load_cuda, cuda_begin
"""

import os
import sys
import subprocess
import hashlib
from pathlib import Path

from torch.utils.cpp_extension import load_inline
from torch.utils import cpp_extension as _cpp_ext

# Standard header block to paste at the top of every cuda_src string.
cuda_begin = r"""
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x)        TORCH_CHECK(x.device().is_cuda(),    #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK(x.is_contiguous(),       #x " must be contiguous")
#define CHECK_INPUT(x)       CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }
"""

_env_ready = False  # set to True only after full compiler env is verified


def _prepend_path(var: str, *dirs: str):
    """Prepend dirs to an env var (PATH, INCLUDE, LIB, etc.) if not already present."""
    current = os.environ.get(var, "")
    new_dirs = [d for d in dirs if d.lower() not in current.lower()]
    if new_dirs:
        os.environ[var] = os.pathsep.join(new_dirs) + (os.pathsep + current if current else "")


def _setup_msvc():
    """
    Set PATH / INCLUDE / LIB / LIBPATH using vswhere + Windows SDK registry key.
    Avoids vcvars64.bat parsing, which is unreliable when launched from Jupyter.
    """
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        raise RuntimeError("vswhere.exe not found — install Visual Studio 2022.")

    # Find latest VS installation that has the C++ tools.
    vs_path_raw = subprocess.run(
        [str(vswhere), "-latest", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True,
    ).stdout.strip()
    if not vs_path_raw:
        raise RuntimeError("No VS installation with C++ tools found via vswhere.")
    vs_path = Path(vs_path_raw)

    # Find latest MSVC toolset version.
    msvc_base = vs_path / "VC" / "Tools" / "MSVC"
    msvc_versions = sorted(msvc_base.iterdir())
    if not msvc_versions:
        raise RuntimeError(f"No MSVC toolset found under {msvc_base}")
    msvc = msvc_versions[-1]

    # Find the Windows 10/11 SDK version from the registry.
    import winreg
    kit_root = sdk_ver = None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Windows Kits\Installed Roots") as key:
            kit_root = Path(winreg.QueryValueEx(key, "KitsRoot10")[0])
    except OSError:
        kit_root = Path(r"C:\Program Files (x86)\Windows Kits\10")
    sdk_include = kit_root / "Include"
    sdk_versions = sorted(sdk_include.iterdir()) if sdk_include.exists() else []
    if sdk_versions:
        sdk_ver = sdk_versions[-1].name

    # --- PATH ---
    _prepend_path("PATH",
        str(msvc / "bin" / "Hostx64" / "x64"),
        str(vs_path / "Common7" / "IDE"),
    )

    # --- INCLUDE ---
    include_dirs = [str(msvc / "include")]
    if sdk_ver:
        for sub in ("ucrt", "um", "shared", "winrt", "cppwinrt"):
            p = kit_root / "Include" / sdk_ver / sub
            if p.exists():
                include_dirs.append(str(p))
    _prepend_path("INCLUDE", *include_dirs)

    # --- LIB ---
    lib_dirs = [str(msvc / "lib" / "x64")]
    if sdk_ver:
        for sub in ("ucrt", "um"):
            p = kit_root / "Lib" / sdk_ver / sub / "x64"
            if p.exists():
                lib_dirs.append(str(p))
    _prepend_path("LIB", *lib_dirs)

    # LIBPATH for MSVC ATL/CRT linkage
    _prepend_path("LIBPATH", str(msvc / "lib" / "x64"))

    # Tell PyTorch that the MSVC env is already set so it skips the
    # broken distutils._msvccompiler path (removed in Python 3.12+).
    os.environ.setdefault("VSCMD_ARG_TGT_ARCH", "x64")


def _setup_env():
    global _env_ready
    if _env_ready:
        return

    # 1. Pin GPU arch so nvcc only compiles for your card (much faster builds).
    import torch as _torch
    _cc = _torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{_cc[0]}.{_cc[1]}")

    # 2. Point PyTorch's JIT compiler at the conda-bundled CUDA toolkit
    #    (nvcc lives at {conda_env}/Library/bin/nvcc.exe).
    #    Also add the conda env's Scripts dir so ninja.exe is on PATH.
    cuda_home = Path(sys.prefix) / "Library"
    if cuda_home.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_home))
        os.environ.setdefault("CUDA_PATH", str(cuda_home))
        _cpp_ext.CUDA_HOME = os.environ["CUDA_HOME"]
        _prepend_path("PATH", str(cuda_home / "bin"), str(Path(sys.prefix) / "Scripts"))

    # 3. On Python 3.8+ Windows, PATH alone doesn't affect DLL loading for .pyd
    #    files — os.add_dll_directory() is required so the compiled extension can
    #    find torch and CUDA DLLs at import time.
    _dll_dirs = [
        Path(_torch.__file__).parent / "lib",  # c10.dll, torch_cuda.dll, …
        cuda_home / "bin",                      # cudart.dll, cublas.dll, …
    ]
    for _d in _dll_dirs:
        if _d.exists():
            os.add_dll_directory(str(_d))

    # 4. Set up the full MSVC compiler environment (cl.exe, INCLUDE, LIB, etc.)
    #    directly via vswhere + registry — no vcvars64.bat parsing needed.
    if "VSCMD_ARG_TGT_ARCH" not in os.environ:
        _setup_msvc()

    _env_ready = True

# load cuda: pebble
# Load inline CUDA extension with caching and unique module name based on source hash.
def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    """Compile and load an inline CUDA extension.

    Args:
        cuda_src: CUDA C++ source string (kernel + torch wrapper).
        cpp_src:  C++ declarations string (one line per exported function).
        funcs:    List of function names to export.
        opt:      Pass -O2 to nvcc when True.
        verbose:  Print ninja build output.

    Returns:
        Loaded extension module.

    Example:
        module = load_cuda(cuda_src, cpp_src, ['my_kernel'])
        out = module.my_kernel(tensor)
    """
    _setup_env()
    cuda_home = Path(sys.prefix) / "Library"
    extra_includes = [str(cuda_home / "include" / "torch" / "csrc" / "api" / "include")]
    extra_ldflags = [f"/LIBPATH:{cuda_home / 'lib'}"]

    # Give each unique source a unique module name.  On Windows, a loaded .pyd
    # is locked for the lifetime of the process, so recompiling to the same
    # filename fails with LNK1104.  A content hash means same code → cache hit
    # (no recompile), changed code → new filename (no lock conflict).
    src_hash = hashlib.md5((cuda_src + cpp_src).encode()).hexdigest()[:8]
    ext_name = f"inline_ext_{src_hash}"
    sys.modules.pop(ext_name, None)

    kwargs = dict(
        cuda_sources=[cuda_src],
        cpp_sources=[cpp_src],
        functions=funcs,
        extra_cuda_cflags=["-O2"] if opt else [],
        extra_include_paths=extra_includes,
        extra_ldflags=extra_ldflags,
        verbose=verbose,
        name=ext_name,
    )

    try:
        return load_inline(**kwargs)
    except ImportError:
        # DLL search paths sometimes need one failed attempt to fully settle
        # on Windows; retry succeeds because the .pyd is already compiled.
        sys.modules.pop(ext_name, None)
        return load_inline(**kwargs)
