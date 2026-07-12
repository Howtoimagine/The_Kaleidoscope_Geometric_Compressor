"""
OptiX RT-Core Recall Backend - the ray-tracing hardware port

Replaces the grid-probe stage of TorchProjectionFilterRecall with real
RT-core BVH traversal (docs/RT_RECALL.md, Stage 1):

    build:  per 24->3 projection, a GAS of axis-aligned bounding boxes,
            one per stored row, side 2*radius around the projected point
    query:  one DEGENERATE ray per projected query (tmin=0, tmax=eps).
            A custom __intersection__ program tests origin-in-sphere
            (the RTNN/TrueKNN encoding); __anyhit__ appends
            (query_id, row_id) to a device buffer with an atomic
            counter and calls optixIgnoreIntersection() so traversal
            continues to every overlapping sphere.
    after:  vote across projections + exact 24-D re-rank (CuPy),
            identical semantics to the CUDA-cores reference.

Requires: pyoptix (optix module), cupy, cuda-python (nvrtc), an RTX GPU,
and driver-side OptiX (ships with GeForce drivers; no SDK needed at
runtime — only headers at compile time).
"""

import ctypes
import os
from typing import List, Optional, Tuple

import numpy as np

# Device programs: compiled once with NVRTC at first use
_CU_SOURCE = r"""
#include <optix.h>

struct Params
{
    OptixTraversableHandle handle;
    const float3*          centers;    // projected stored rows
    const float3*          queries;    // projected query points
    unsigned long long*    hits;       // packed (query_id<<32 | row_id)
    unsigned int*          hit_count;
    unsigned int           max_hits;
    float                  radius_sq;
};

extern "C" { __constant__ Params params; }

extern "C" __global__ void __raygen__rg()
{
    const unsigned int qid = optixGetLaunchIndex().x;
    const float3 q = params.queries[qid];

    unsigned int p0 = qid;
    optixTrace(
        params.handle,
        q,                          // origin = the query point
        make_float3(1.f, 0.f, 0.f), // direction irrelevant:
        0.0f,                       // tmin
        1.0e-16f,                   // tmax ~ 0 -> containment test only
        0.0f,                       // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,                    // SBT offset/stride/miss index
        p0);
}

extern "C" __global__ void __intersection__contains()
{
    const unsigned int rid = optixGetPrimitiveIndex();
    const float3 q = optixGetWorldRayOrigin();
    const float3 c = params.centers[rid];
    const float dx = q.x - c.x, dy = q.y - c.y, dz = q.z - c.z;
    if (dx * dx + dy * dy + dz * dz <= params.radius_sq)
        optixReportIntersection(0.0f, 0);
}

extern "C" __global__ void __anyhit__record()
{
    const unsigned int qid = optixGetPayload_0();
    const unsigned int rid = optixGetPrimitiveIndex();
    const unsigned int slot = atomicAdd(params.hit_count, 1u);
    if (slot < params.max_hits)
        params.hits[slot] =
            (static_cast<unsigned long long>(qid) << 32) | rid;
    optixIgnoreIntersection();   // keep traversing: we want EVERY sphere
}

extern "C" __global__ void __miss__ms() {}
"""


def _import_stack():
    try:
        import cupy as cp
        import optix
        from cuda.bindings import nvrtc
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "OptiX backend needs pyoptix + cupy + cuda-python "
            "(see docs/RT_RECALL.md, Practical stack)"
        ) from e
    return cp, optix, nvrtc


def _cuda_include_dir() -> str:
    cuda = os.environ.get("CUDA_PATH")
    if cuda and os.path.isdir(os.path.join(cuda, "include")):
        return os.path.join(cuda, "include")
    root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(root):
        for v in sorted(os.listdir(root), reverse=True):
            inc = os.path.join(root, v, "include")
            if os.path.isdir(inc):
                return inc
    raise RuntimeError("CUDA toolkit include directory not found")


def _optix_include_dir() -> str:
    for cand in (
        os.environ.get("OPTIX_INCLUDE_PATH"),
        os.path.expanduser(r"~\sdks\optix-dev\include"),
    ):
        if cand and os.path.isfile(os.path.join(cand, "optix.h")):
            return cand
    raise RuntimeError(
        "OptiX headers not found: set OPTIX_INCLUDE_PATH or clone "
        "https://github.com/NVIDIA/optix-dev to ~/sdks/optix-dev"
    )


def _compile_ptx() -> bytes:
    _, _, nvrtc = _import_stack()

    def check(result, prog=None):
        if result[0].value:
            msg = nvrtc.nvrtcGetErrorString(result[0])[1]
            if prog is not None:
                _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
                log = b" " * log_size
                nvrtc.nvrtcGetProgramLog(prog, log)
                msg = f"{msg}: {log.decode(errors='replace')}"
            raise RuntimeError(f"NVRTC: {msg}")
        return result[1] if len(result) == 2 else result[1:]

    opts = [
        b"-use_fast_math",
        b"-default-device",
        b"-std=c++17",
        b"-rdc",
        b"true",
        f"-I{_optix_include_dir()}".encode(),
        f"-I{_cuda_include_dir()}".encode(),
    ]
    prog = check(
        nvrtc.nvrtcCreateProgram(_CU_SOURCE.encode(), b"rt_recall.cu", 0, [], [])
    )
    check(nvrtc.nvrtcCompileProgram(prog, len(opts), opts), prog)
    (size,) = (check(nvrtc.nvrtcGetPTXSize(prog)),)
    ptx = b" " * size
    check(nvrtc.nvrtcGetPTX(prog, ptx))
    return ptx


def _round_up(v: int, m: int) -> int:
    return v if v % m == 0 else v + m - v % m


class _OptixEngine:
    """One device context + pipeline, shared across all projections."""

    _instance: Optional["_OptixEngine"] = None

    @classmethod
    def get(cls) -> "_OptixEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        cp, optix, _ = _import_stack()
        self.cp = cp
        self.optix = optix
        cp.cuda.runtime.free(0)  # force CUDA context creation

        self.ctx = optix.deviceContextCreate(
            0, optix.DeviceContextOptions(logCallbackLevel=2)
        )

        pipeline_opts = optix.PipelineCompileOptions(
            usesMotionBlur=False,
            traversableGraphFlags=int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
            numPayloadValues=1,
            numAttributeValues=2,
            exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
            pipelineLaunchParamsVariableName="params",
            usesPrimitiveTypeFlags=int(optix.PRIMITIVE_TYPE_FLAGS_CUSTOM),
        )

        module_opts = optix.ModuleCompileOptions(
            maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )
        module, _log = self.ctx.moduleCreate(module_opts, pipeline_opts, _compile_ptx())

        rg = optix.ProgramGroupDesc()
        rg.raygenModule = module
        rg.raygenEntryFunctionName = "__raygen__rg"
        ms = optix.ProgramGroupDesc()
        ms.missModule = module
        ms.missEntryFunctionName = "__miss__ms"
        hg = optix.ProgramGroupDesc()
        hg.hitgroupModuleIS = module
        hg.hitgroupEntryFunctionNameIS = "__intersection__contains"
        hg.hitgroupModuleAH = module
        hg.hitgroupEntryFunctionNameAH = "__anyhit__record"
        groups = []
        for desc in (rg, ms, hg):
            pg, _log = self.ctx.programGroupCreate([desc])
            groups.append(pg[0])
        self.groups = groups

        link_opts = optix.PipelineLinkOptions()
        link_opts.maxTraceDepth = 1
        self.pipeline = self.ctx.pipelineCreate(
            pipeline_opts, link_opts, groups, ""
        )
        stack_sizes = optix.StackSizes()
        for g in groups:
            optix.util.accumulateStackSizes(g, stack_sizes, self.pipeline)
        sizes = optix.util.computeStackSizes(stack_sizes, 1, 0, 0)
        self.pipeline.setStackSize(*sizes, 1)

        self.sbt = self._make_sbt()

    def _sbt_record(self, group) -> "object":
        cp, optix = self.cp, self.optix
        fmt = "{}B".format(optix.SBT_RECORD_HEADER_SIZE)
        dtype = np.dtype(
            {
                "names": ["header"],
                "formats": [fmt],
                "itemsize": _round_up(
                    optix.SBT_RECORD_HEADER_SIZE, optix.SBT_RECORD_ALIGNMENT
                ),
                "align": True,
            }
        )
        rec = np.zeros(1, dtype=dtype)
        optix.sbtRecordPackHeader(group, rec)
        d = cp.cuda.memory.alloc(rec.nbytes)
        d.copy_from(ctypes.c_void_p(rec.ctypes.data), rec.nbytes)
        return d, rec.nbytes

    def _make_sbt(self):
        optix = self.optix
        self._d_rg, _ = self._sbt_record(self.groups[0])
        self._d_ms, ms_size = self._sbt_record(self.groups[1])
        self._d_hg, hg_size = self._sbt_record(self.groups[2])
        return optix.ShaderBindingTable(
            raygenRecord=self._d_rg.ptr,
            missRecordBase=self._d_ms.ptr,
            missRecordStrideInBytes=ms_size,
            missRecordCount=1,
            hitgroupRecordBase=self._d_hg.ptr,
            hitgroupRecordStrideInBytes=hg_size,
            hitgroupRecordCount=1,
        )

    def build_gas(self, centers_low: "object", radius: float):
        """GAS of one AABB per projected row (centers_low: (N,3) cupy f4)."""
        cp, optix = self.cp, self.optix
        n = centers_low.shape[0]
        lo = centers_low - radius
        hi = centers_low + radius
        aabbs = cp.concatenate([lo, hi], axis=1).astype(cp.float32).ravel()

        build_input = optix.BuildInputCustomPrimitiveArray(
            aabbBuffers=[aabbs.data.ptr],
            numPrimitives=n,
            flags=[optix.GEOMETRY_FLAG_NONE],
            numSbtRecords=1,
        )
        accel_opts = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_PREFER_FAST_TRACE),
            operation=optix.BUILD_OPERATION_BUILD,
        )
        sizes = self.ctx.accelComputeMemoryUsage([accel_opts], [build_input])
        d_temp = cp.cuda.alloc(sizes.tempSizeInBytes)
        d_out = cp.cuda.alloc(sizes.outputSizeInBytes)
        handle = self.ctx.accelBuild(
            0,
            [accel_opts],
            [build_input],
            d_temp.ptr,
            sizes.tempSizeInBytes,
            d_out.ptr,
            sizes.outputSizeInBytes,
            [],
        )
        return handle, d_out  # d_out must stay alive with the handle

    def launch(
        self,
        handle,
        centers: "object",
        queries_low: "object",
        radius: float,
        max_hits: int,
    ) -> Tuple["object", int]:
        """Fire one degenerate ray per query; return packed hit pairs."""
        cp, optix = self.cp, self.optix
        nq = queries_low.shape[0]
        d_hits = cp.zeros(max_hits, dtype=cp.uint64)
        d_count = cp.zeros(1, dtype=cp.uint32)

        params_dtype = np.dtype(
            {
                "names": [
                    "handle", "centers", "queries", "hits",
                    "hit_count", "max_hits", "radius_sq",
                ],
                "formats": ["u8", "u8", "u8", "u8", "u8", "u4", "f4"],
                "itemsize": _round_up(8 * 5 + 4 + 4, 8),
                "align": True,
            }
        )
        h_params = np.array(
            [(
                handle,
                centers.data.ptr,
                queries_low.data.ptr,
                d_hits.data.ptr,
                d_count.data.ptr,
                max_hits,
                np.float32(radius**2),
            )],
            dtype=params_dtype,
        )
        d_params = cp.cuda.memory.alloc(h_params.nbytes)
        d_params.copy_from(ctypes.c_void_p(h_params.ctypes.data), h_params.nbytes)

        stream = cp.cuda.Stream()
        optix.launch(
            self.pipeline,
            stream.ptr,
            d_params.ptr,
            h_params.dtype.itemsize,
            self.sbt,
            nq,
            1,
            1,
        )
        stream.synchronize()
        n_hits = int(d_count.get()[0])
        if n_hits > max_hits:
            # The first pass still gives the exact total through d_count;
            # its output buffer is truncated, so replay once at that size.
            # Dense valid neighborhoods must not turn into a benchmark crash
            # or silently lose candidates.
            return self.launch(handle, centers, queries_low, radius, n_hits)
        return d_hits[:n_hits], n_hits


class OptixProjectionFilterRecall:
    """
    Recall-as-rays on actual RT cores.

    Same pipeline and parameters as TorchProjectionFilterRecall — the
    grid probe is replaced by BVH traversal on RT hardware; vote and
    exact re-rank run in CuPy. Identical seeded projections, so results
    are comparable across the three backends.
    """

    def __init__(
        self,
        rows: np.ndarray,
        n_projections: int = 4,
        votes: int = 2,
        radius: float = 1.5,
        seed: int = 7,
        max_hits_per_query: int = 512,
        query_chunk: int = 4096,
    ) -> None:
        cp, _, _ = _import_stack()
        self.cp = cp
        self.votes = votes
        self.radius = radius
        self.max_hits_per_query = max_hits_per_query
        self.query_chunk = query_chunk
        rows = np.asarray(rows, dtype=np.float32)
        self.n = rows.shape[0]
        self.rows = cp.asarray(rows)
        self.row_sq = (self.rows**2).sum(axis=1)

        rng = np.random.default_rng(seed)
        projs = []
        for _ in range(n_projections):
            a = rng.normal(size=(24, 3))
            q, _ = np.linalg.qr(a)
            projs.append(q * np.sqrt(24.0 / 3.0))
        self.projs = cp.asarray(np.stack(projs).astype(np.float32))  # (m,24,3)

        self.engine = _OptixEngine.get()
        self.low: List = []
        self.gas: List = []
        for i in range(n_projections):
            low = (self.rows @ self.projs[i]).astype(cp.float32)
            low = cp.ascontiguousarray(low)
            self.low.append(low)
            self.gas.append(self.engine.build_gas(low, radius))  # (handle, buf)

    def query(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        qnp = np.atleast_2d(np.asarray(q, dtype=np.float32))
        nq = qnp.shape[0]
        if nq <= self.query_chunk:
            return self._query_chunk(qnp, k)
        idx = np.empty((nq, k), dtype=np.int64)
        dist = np.empty((nq, k), dtype=np.float64)
        for s in range(0, nq, self.query_chunk):
            e = min(nq, s + self.query_chunk)
            idx[s:e], dist[s:e] = self._query_chunk(qnp[s:e], k)
        return idx, dist

    def _query_chunk(self, qnp: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        cp = self.cp
        qt = cp.asarray(qnp)
        nq = qt.shape[0]
        max_hits = self.max_hits_per_query * nq

        pair_list = []
        for i in range(len(self.gas)):
            ql = cp.ascontiguousarray((qt @ self.projs[i]).astype(cp.float32))
            handle, _buf = self.gas[i]
            hits, n_hits = self.engine.launch(
                handle, self.low[i], ql, self.radius, max_hits
            )
            if n_hits:
                pair_list.append(hits)

        idx_out = np.full((nq, k), -1, dtype=np.int64)
        d_out = np.full((nq, k), np.inf)
        if not pair_list:
            return idx_out, d_out

        pairs = cp.concatenate(pair_list)
        pairs, counts = cp.unique(pairs, return_counts=True)
        pairs = pairs[counts >= self.votes]
        if not int(pairs.size):
            return idx_out, d_out
        qid = (pairs >> 32).astype(cp.int64)
        rowid = (pairs & cp.uint64(0xFFFFFFFF)).astype(cp.int64)

        d2 = (
            self.row_sq[rowid]
            - 2.0 * (self.rows[rowid] * qt[qid]).sum(axis=1)
            + (qt[qid] ** 2).sum(axis=1)
        )
        d2 = cp.maximum(d2, 0.0)

        order = cp.argsort(d2)
        qid, rowid, d2 = qid[order], rowid[order], d2[order]
        order = cp.argsort(qid, kind="stable")
        qid, rowid, d2 = qid[order], rowid[order], d2[order]

        qid_np = cp.asnumpy(qid)
        rowid_np = cp.asnumpy(rowid)
        d2_np = cp.asnumpy(d2).astype(np.float64)
        bounds = np.searchsorted(qid_np, np.arange(nq + 1))
        for j in range(nq):
            s, e = bounds[j], min(bounds[j + 1], bounds[j] + k)
            m = e - s
            idx_out[j, :m] = rowid_np[s:e]
            d_out[j, :m] = d2_np[s:e]
        return idx_out, d_out
