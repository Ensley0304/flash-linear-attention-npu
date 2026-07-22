#!/usr/bin/env bash
# Copyright (c) 2026 Tianjin University, Ltd.

# Reproducible build, precision and profiling flow for grouped KDA backward.
# Run this script from an already activated Python/torch_npu environment.

set -Eeuo pipefail

MODE="build"
SOURCE_ROOT=""
CANN_ENV=""
STATE_FILE=""
SOC="ascend910b"
PHYSICAL_DEVICE="2"
RUN_BASE="/var/tmp"
JOBS="8"
FULL_METRICS=false
FINAL_GATE=false
PAIR_GATES_MODE="source"
SHARED_SETUP_MODE="source"
STAGE_EPILOGUE_MODE="source"
PAIR_SCRATCH_MODE="source"
TAIL_BLOCKS_MODE="source"
TASK_STORE_MODE="source"
MMAD_ENGINES_MODE="source"
VECTOR_MASK_MODE="source"
DB_REDUCE_MODE="source"
STAGE_A_MODE="source"
CUBE_MODE="source"
STAGE_IO_MODE="source"
AIC_DIAGNOSTIC_MODE="source"
EXPECTED_CATLASS_COMMIT="41bf90da655bba3c66d0acd7e00abe33960ecfd6"
VALIDATION_SOURCES=(
    fla/ops/ascendc/common/kernel_utils/block/block_mmad_pingpong_tla_multi.hpp
    fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_host/chunk_kda_bwd_intra_tiling.cpp
    fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra.cpp
    fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra_grouped.hpp
    fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra_mixed.hpp
    tests/reference/chunk_kda_reference.py
    torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py
    torch_custom/fla_npu/test/benchmark_npu_chunk_kda_bwd_intra.py
    scripts/analyze_chunk_kda_bwd_intra_profile.py
    scripts/compare_chunk_kda_bwd_intra_profiles.py
    scripts/model_chunk_kda_bwd_intra_blockdiag.py
    scripts/model_chunk_kda_bwd_intra_traffic.py
    scripts/run_chunk_kda_bwd_intra_grouped_validation.sh
)

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh [options]

Options:
  --mode build|test|profile|all  Default: build
  --source PATH                  Repository root; default: parent of this script
  --cann-env PATH                Required CANN set_env.sh
  --state PATH                   state.env produced by build (test/profile mode)
  --soc NAME                     Default: ascend910b
  --physical-device N            Physical device exposed to the process; default: 2
  --run-base PATH                Persistent artifact parent; default: /var/tmp
  --jobs N                       build.sh parallelism; default: 8
  --pair-gates MODE              source|factor|direct; build/all only
  --shared-setup MODE            source|overlap|prologue; build/all only
  --stage-epilogue MODE          source|overlap|tail; build/all only
  --pair-scratch MODE            source|pingpong|single; build/all only
  --tail-blocks MODE             source|batch|scalar; build/all only
  --task-store MODE              source|overlap|serial; build/all only
  --mmad-engines MODE            source|persistent|scoped; build/all only
  --vector-mask MODE             source|reuse|per-call; build/all only
  --db-reduce MODE               source|coalesced|per-row; build/all only
  --stage-a MODE                 source|packed|split; build/all only
  --cube-mode MODE               source|ieee|hf32; build/all only
  --stage-io MODE                source|tscm|gm; build/all only. tscm is rejected on ascend910b
  --aic-diagnostic MODE          source|full|handshake|stage0-right|stage0-left|
                                 stage0-both|through-stage1|through-stage2;
                                 build only. Partial modes are timeout diagnostics,
                                 never precision/performance candidates
  --full-metrics                 Collect all seven msprof metric groups + sample mode
  --final-gate                   Make profile fail unless median kernel time is <= 4 ms
  -h, --help

Examples:
  bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
    --mode build --cann-env /path/to/ascend-toolkit/set_env.sh

  bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
    --mode test --state /var/tmp/wys_kda_grouped_*/state.env \
    --cann-env /path/to/ascend-toolkit/set_env.sh

  # Build/test/profile an isolated direct-gate A/B variant of the same commit.
  bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
    --mode all --cann-env /path/to/ascend-toolkit/set_env.sh \
    --pair-gates direct --shared-setup overlap --stage-epilogue tail \
    --pair-scratch single --tail-blocks batch --task-store serial --mmad-engines persistent \
    --vector-mask reuse --db-reduce coalesced --stage-a split --cube-mode ieee \
    --stage-io gm
EOF
}

require_option_value() {
    [[ $# -ge 2 && -n "${2:-}" && "${2:-}" != --* ]] || {
        echo "[FAIL] option $1 requires a value" >&2
        exit 2
    }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) require_option_value "$@"; MODE="$2"; shift 2 ;;
        --source) require_option_value "$@"; SOURCE_ROOT="$2"; shift 2 ;;
        --cann-env) require_option_value "$@"; CANN_ENV="$2"; shift 2 ;;
        --state) require_option_value "$@"; STATE_FILE="$2"; shift 2 ;;
        --soc) require_option_value "$@"; SOC="$2"; shift 2 ;;
        --physical-device) require_option_value "$@"; PHYSICAL_DEVICE="$2"; shift 2 ;;
        --run-base) require_option_value "$@"; RUN_BASE="$2"; shift 2 ;;
        --jobs) require_option_value "$@"; JOBS="$2"; shift 2 ;;
        --pair-gates) require_option_value "$@"; PAIR_GATES_MODE="$2"; shift 2 ;;
        --shared-setup) require_option_value "$@"; SHARED_SETUP_MODE="$2"; shift 2 ;;
        --stage-epilogue) require_option_value "$@"; STAGE_EPILOGUE_MODE="$2"; shift 2 ;;
        --pair-scratch) require_option_value "$@"; PAIR_SCRATCH_MODE="$2"; shift 2 ;;
        --tail-blocks) require_option_value "$@"; TAIL_BLOCKS_MODE="$2"; shift 2 ;;
        --task-store) require_option_value "$@"; TASK_STORE_MODE="$2"; shift 2 ;;
        --mmad-engines) require_option_value "$@"; MMAD_ENGINES_MODE="$2"; shift 2 ;;
        --vector-mask) require_option_value "$@"; VECTOR_MASK_MODE="$2"; shift 2 ;;
        --db-reduce) require_option_value "$@"; DB_REDUCE_MODE="$2"; shift 2 ;;
        --stage-a) require_option_value "$@"; STAGE_A_MODE="$2"; shift 2 ;;
        --cube-mode) require_option_value "$@"; CUBE_MODE="$2"; shift 2 ;;
        --stage-io) require_option_value "$@"; STAGE_IO_MODE="$2"; shift 2 ;;
        --aic-diagnostic) require_option_value "$@"; AIC_DIAGNOSTIC_MODE="$2"; shift 2 ;;
        --full-metrics) FULL_METRICS=true; shift ;;
        --final-gate) FINAL_GATE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[FAIL] unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

case "$MODE" in
    build|test|profile|all) ;;
    *) echo "[FAIL] invalid mode: $MODE" >&2; usage; exit 2 ;;
esac
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[FAIL] --jobs must be positive" >&2; exit 2; }
[[ "$PHYSICAL_DEVICE" =~ ^[0-9]+$ ]] || {
    echo "[FAIL] --physical-device must be a non-negative integer" >&2
    exit 2
}
case "$PAIR_GATES_MODE" in
    source|factor|direct) ;;
    *) echo "[FAIL] --pair-gates must be source, factor or direct" >&2; exit 2 ;;
esac
case "$SHARED_SETUP_MODE" in
    source|overlap|prologue) ;;
    *) echo "[FAIL] --shared-setup must be source, overlap or prologue" >&2; exit 2 ;;
esac
case "$STAGE_EPILOGUE_MODE" in
    source|overlap|tail) ;;
    *) echo "[FAIL] --stage-epilogue must be source, overlap or tail" >&2; exit 2 ;;
esac
case "$PAIR_SCRATCH_MODE" in
    source|pingpong|single) ;;
    *) echo "[FAIL] --pair-scratch must be source, pingpong or single" >&2; exit 2 ;;
esac
case "$TAIL_BLOCKS_MODE" in
    source|batch|scalar) ;;
    *) echo "[FAIL] --tail-blocks must be source, batch or scalar" >&2; exit 2 ;;
esac
case "$TASK_STORE_MODE" in
    source|overlap|serial) ;;
    *) echo "[FAIL] --task-store must be source, overlap or serial" >&2; exit 2 ;;
esac
case "$MMAD_ENGINES_MODE" in
    source|persistent|scoped) ;;
    *) echo "[FAIL] --mmad-engines must be source, persistent or scoped" >&2; exit 2 ;;
esac
case "$VECTOR_MASK_MODE" in
    source|reuse|per-call) ;;
    *) echo "[FAIL] --vector-mask must be source, reuse or per-call" >&2; exit 2 ;;
esac
case "$DB_REDUCE_MODE" in
    source|coalesced|per-row) ;;
    *) echo "[FAIL] --db-reduce must be source, coalesced or per-row" >&2; exit 2 ;;
esac
case "$STAGE_A_MODE" in
    source|packed|split) ;;
    *) echo "[FAIL] --stage-a must be source, packed or split" >&2; exit 2 ;;
esac
case "$CUBE_MODE" in
    source|ieee|hf32) ;;
    *) echo "[FAIL] --cube-mode must be source, ieee or hf32" >&2; exit 2 ;;
esac
case "$STAGE_IO_MODE" in
    source|tscm|gm) ;;
    *) echo "[FAIL] --stage-io must be source, tscm or gm" >&2; exit 2 ;;
esac
case "$AIC_DIAGNOSTIC_MODE" in
    source|full|handshake|stage0-right|stage0-left|stage0-both|through-stage1|through-stage2) ;;
    *) echo "[FAIL] invalid --aic-diagnostic mode: $AIC_DIAGNOSTIC_MODE" >&2; exit 2 ;;
esac
if [[ "$MODE" == test || "$MODE" == profile ]]; then
    [[ "$PAIR_GATES_MODE" == source && "$SHARED_SETUP_MODE" == source && \
       "$STAGE_EPILOGUE_MODE" == source && "$PAIR_SCRATCH_MODE" == source && \
       "$TAIL_BLOCKS_MODE" == source && "$TASK_STORE_MODE" == source && \
       "$MMAD_ENGINES_MODE" == source && \
       "$VECTOR_MASK_MODE" == source && "$DB_REDUCE_MODE" == source && \
       "$STAGE_A_MODE" == source && "$CUBE_MODE" == source && \
       "$STAGE_IO_MODE" == source && "$AIC_DIAGNOSTIC_MODE" == source ]] || {
        echo "[FAIL] test/profile variants come from --state; do not override them" >&2
        exit 2
    }
fi
if [[ "$AIC_DIAGNOSTIC_MODE" != source && "$AIC_DIAGNOSTIC_MODE" != full && \
      "$MODE" != build ]]; then
    echo "[FAIL] partial --aic-diagnostic modes are build-only" >&2
    exit 2
fi
if $FINAL_GATE; then
    [[ "$MODE" == profile || "$MODE" == all ]] || {
        echo "[FAIL] --final-gate is valid only for profile/all mode" >&2
        exit 2
    }
    $FULL_METRICS || {
        echo "[FAIL] --final-gate requires --full-metrics" >&2
        exit 2
    }
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RUNNER_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -z "$SOURCE_ROOT" ]]; then
    SOURCE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
else
    SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd -P)"
fi

[[ -n "$CANN_ENV" ]] || {
    echo "[FAIL] --cann-env is required; do not rely on /usr/local CANN aliases" >&2
    exit 2
}
[[ -f "$CANN_ENV" ]] || { echo "[FAIL] missing CANN env: $CANN_ENV" >&2; exit 2; }
CANN_ENV="$(cd "$(dirname "$CANN_ENV")" && pwd -P)/$(basename "$CANN_ENV")"
mkdir -p "$RUN_BASE"
RUN_BASE="$(cd "$RUN_BASE" && pwd -P)"

source_cann() {
    local rc
    set +e
    set +u
    unset PYTHONPATH LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH \
        ASCEND_AICPU_PATH ASCEND_CUSTOM_OPP_PATH FLA_NPU_OP_API_LIB TILING_KEY
    # shellcheck disable=SC1090
    source "$CANN_ENV"
    rc=$?
    set -u
    set -e
    [[ $rc -eq 0 ]] || { echo "[FAIL] source CANN rc=$rc" >&2; exit "$rc"; }
    export PYTHONNOUSERSITE=1
    export FLA_NPU_SOC="$SOC"
    unset FLA_NPU_OPS FLA_NPU_SKIP_RUN_BUILD FLA_NPU_SKIP_RUN_INSTALL \
        FLA_NPU_INCREMENTAL_BUILD FLA_NPU_BUILD_LEGACY_EXTENSION TILING_KEY
    hash -r

    [[ -n "${ASCEND_HOME_PATH:-}" ]] || {
        echo "[FAIL] CANN env did not set ASCEND_HOME_PATH" >&2
        exit 1
    }
    CURRENT_CANN_ENV="$(readlink -f "$CANN_ENV")"
    CURRENT_CANN_ENV_SHA256="$(sha256sum "$CURRENT_CANN_ENV" | awk '{print $1}')"
    CURRENT_ASCEND_HOME="$(readlink -f "$ASCEND_HOME_PATH")"
    CURRENT_PYTHON="$(readlink -f "$(command -v python3)")"
    CURRENT_PYTHON_VERSION="$(python3 -c 'import platform; print(platform.python_version())')"
    CURRENT_TORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__)')"
    CURRENT_TORCH_NPU_VERSION="$(python3 -c 'import torch_npu; print(torch_npu.__version__)')"

    local cann_bundle_root
    cann_bundle_root="$(cd "$(dirname "$CANN_ENV")/.." && pwd -P)"
    case "$CURRENT_ASCEND_HOME/" in
        "$cann_bundle_root/"*) ;;
        *)
            echo "[FAIL] ASCEND_HOME_PATH is outside the selected CANN bundle" >&2
            echo "       env=$CURRENT_CANN_ENV" >&2
            echo "       home=$CURRENT_ASCEND_HOME" >&2
            exit 1
            ;;
    esac
    local -a cann_version_files=()
    mapfile -t cann_version_files < <(find "$CURRENT_ASCEND_HOME" -maxdepth 3 -type f \
        \( -name 'version.info' -o -name 'version.cfg' -o -name 'version.txt' \) \
        -print | sort)
    if [[ ${#cann_version_files[@]} -gt 0 ]]; then
        CURRENT_CANN_VERSION_DIGEST="$(sha256sum "${cann_version_files[@]}" | \
            awk '{print $1}' | sort | sha256sum | awk '{print $1}')"
    else
        CURRENT_CANN_VERSION_DIGEST='<no-version-file>'
    fi
}

write_state() {
    local state=$1 name
    : >"$state"
    for name in RUN_ROOT WHEEL_SRC WHEEL WHEEL_INSTALL COMMIT WHEEL_SHA256 \
        WHEEL_KDA_DIGEST BUILD_CANN_ENV BUILD_CANN_ENV_SHA256 \
        BUILD_ASCEND_HOME BUILD_CANN_VERSION_DIGEST BUILD_SOC \
        BUILD_PYTHON BUILD_PYTHON_VERSION BUILD_TORCH_VERSION \
        BUILD_TORCH_NPU_VERSION VALIDATION_MANIFEST \
        VALIDATION_MANIFEST_SHA256 BUILD_CATLASS_COMMIT BUILD_CATLASS_TREE \
        BUILD_RUNNER_SHA256 BUILD_PAIR_GATES BUILD_SHARED_SETUP \
        BUILD_STAGE_EPILOGUE BUILD_PAIR_SCRATCH BUILD_TAIL_BLOCKS BUILD_TASK_STORE \
        BUILD_MMAD_ENGINES BUILD_VECTOR_MASK BUILD_DB_REDUCE BUILD_STAGE_A BUILD_CUBE_MODE \
        BUILD_STAGE_IO BUILD_AIC_DIAGNOSTIC \
        BUILD_VARIANT_ID; do
        printf 'export %s=%q\n' "$name" "${!name}" >>"$state"
    done
}

load_state() {
    [[ -n "$STATE_FILE" ]] || {
        echo "[FAIL] --state is required for mode $MODE" >&2
        exit 2
    }
    [[ -f "$STATE_FILE" ]] || { echo "[FAIL] state not found: $STATE_FILE" >&2; exit 2; }
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    for required in RUN_ROOT WHEEL_SRC WHEEL WHEEL_INSTALL COMMIT WHEEL_SHA256 \
        WHEEL_KDA_DIGEST BUILD_CANN_ENV BUILD_CANN_ENV_SHA256 \
        BUILD_ASCEND_HOME BUILD_CANN_VERSION_DIGEST BUILD_SOC \
        BUILD_PYTHON BUILD_PYTHON_VERSION BUILD_TORCH_VERSION \
        BUILD_TORCH_NPU_VERSION VALIDATION_MANIFEST \
        VALIDATION_MANIFEST_SHA256 BUILD_CATLASS_COMMIT BUILD_CATLASS_TREE \
        BUILD_RUNNER_SHA256 BUILD_PAIR_GATES BUILD_SHARED_SETUP \
        BUILD_STAGE_EPILOGUE BUILD_PAIR_SCRATCH BUILD_TAIL_BLOCKS BUILD_TASK_STORE \
        BUILD_MMAD_ENGINES BUILD_VECTOR_MASK BUILD_DB_REDUCE BUILD_STAGE_A BUILD_CUBE_MODE \
        BUILD_STAGE_IO BUILD_AIC_DIAGNOSTIC \
        BUILD_VARIANT_ID; do
        [[ -n "${!required:-}" ]] || { echo "[FAIL] state misses $required" >&2; exit 2; }
    done
    [[ -f "$WHEEL" ]] || { echo "[FAIL] wheel disappeared: $WHEEL" >&2; exit 2; }
    [[ -d "$WHEEL_SRC" ]] || { echo "[FAIL] wheel source disappeared: $WHEEL_SRC" >&2; exit 2; }
    [[ -f "$VALIDATION_MANIFEST" ]] || {
        echo "[FAIL] validation manifest disappeared: $VALIDATION_MANIFEST" >&2
        exit 2
    }
    case "$BUILD_PAIR_GATES" in factor|direct) ;; *)
        echo "[FAIL] invalid pair-gates value in state: $BUILD_PAIR_GATES" >&2
        exit 2
    esac
    case "$BUILD_SHARED_SETUP" in overlap|prologue) ;; *)
        echo "[FAIL] invalid shared-setup value in state: $BUILD_SHARED_SETUP" >&2
        exit 2
    esac
    case "$BUILD_STAGE_EPILOGUE" in overlap|tail) ;; *)
        echo "[FAIL] invalid stage-epilogue value in state: $BUILD_STAGE_EPILOGUE" >&2
        exit 2
    esac
    case "$BUILD_PAIR_SCRATCH" in pingpong|single) ;; *)
        echo "[FAIL] invalid pair-scratch value in state: $BUILD_PAIR_SCRATCH" >&2
        exit 2
    esac
    case "$BUILD_TAIL_BLOCKS" in batch|scalar) ;; *)
        echo "[FAIL] invalid tail-blocks value in state: $BUILD_TAIL_BLOCKS" >&2
        exit 2
    esac
    case "$BUILD_TASK_STORE" in overlap|serial) ;; *)
        echo "[FAIL] invalid task-store value in state: $BUILD_TASK_STORE" >&2
        exit 2
    esac
    case "$BUILD_MMAD_ENGINES" in persistent|scoped) ;; *)
        echo "[FAIL] invalid MMAD-engines value in state: $BUILD_MMAD_ENGINES" >&2
        exit 2
    esac
    case "$BUILD_VECTOR_MASK" in reuse|per-call) ;; *)
        echo "[FAIL] invalid Vector-mask value in state: $BUILD_VECTOR_MASK" >&2
        exit 2
    esac
    case "$BUILD_DB_REDUCE" in coalesced|per-row) ;; *)
        echo "[FAIL] invalid db-reduce value in state: $BUILD_DB_REDUCE" >&2
        exit 2
    esac
    case "$BUILD_STAGE_A" in packed|split) ;; *)
        echo "[FAIL] invalid stage-A value in state: $BUILD_STAGE_A" >&2
        exit 2
    esac
    case "$BUILD_CUBE_MODE" in ieee|hf32) ;; *)
        echo "[FAIL] invalid Cube mode in state: $BUILD_CUBE_MODE" >&2
        exit 2
    esac
    case "$BUILD_STAGE_IO" in tscm|gm) ;; *)
        echo "[FAIL] invalid stage-I/O value in state: $BUILD_STAGE_IO" >&2
        exit 2
    esac
    case "$BUILD_AIC_DIAGNOSTIC" in full|handshake|stage0-right|stage0-left|stage0-both|through-stage1|through-stage2) ;; *)
        echo "[FAIL] invalid AIC diagnostic value in state: $BUILD_AIC_DIAGNOSTIC" >&2
        exit 2
    esac
    if [[ "$BUILD_STAGE_IO" == tscm && "$BUILD_SOC" == ascend910b ]]; then
        echo "[FAIL] refusing unsupported A2 TSCM stage-I/O artifact from state" >&2
        echo "       rebuild with --stage-io gm" >&2
        exit 2
    fi
    if [[ "$BUILD_TASK_STORE" == overlap ]]; then
        [[ "$BUILD_TAIL_BLOCKS" == batch && "$BUILD_STAGE_EPILOGUE" == tail ]] || {
            echo "[FAIL] task-store overlap requires tail-blocks=batch and stage-epilogue=tail" >&2
            exit 2
        }
    fi
    local expected_variant_id
    expected_variant_id="pair-${BUILD_PAIR_GATES}_setup-${BUILD_SHARED_SETUP}_epilogue-${BUILD_STAGE_EPILOGUE}_scratch-${BUILD_PAIR_SCRATCH}_tail-${BUILD_TAIL_BLOCKS}_mmad-${BUILD_MMAD_ENGINES}_vmask-${BUILD_VECTOR_MASK}_dbr-${BUILD_DB_REDUCE}_store-${BUILD_TASK_STORE}_stagea-${BUILD_STAGE_A}_cube-${BUILD_CUBE_MODE}_io-${BUILD_STAGE_IO}"
    if [[ "$BUILD_AIC_DIAGNOSTIC" != full ]]; then
        expected_variant_id+="_aicdiag-${BUILD_AIC_DIAGNOSTIC}"
    fi
    [[ "$BUILD_VARIANT_ID" == "$expected_variant_id" ]] || {
        echo "[FAIL] build variant id is inconsistent with state" >&2
        exit 2
    }
    if [[ "$MODE" != build && "$BUILD_AIC_DIAGNOSTIC" != full ]]; then
        echo "[FAIL] partial AIC diagnostic artifacts are build-only" >&2
        exit 2
    fi
}

verify_clean_catlass() {
    local purpose=$1
    local catlass_root="$RUN_ROOT/third_party/catlass"
    local status
    git -C "$catlass_root" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        echo "[FAIL] $purpose CATLASS checkout is missing" >&2
        return 1
    }
    status="$(git -C "$catlass_root" status --porcelain=v1 \
        --untracked-files=all --ignore-submodules=none)"
    [[ -z "$status" ]] || {
        echo "[FAIL] $purpose CATLASS checkout is dirty" >&2
        printf '%s\n' "$status" >&2
        return 1
    }
}

cmake_value_is_false() {
    local normalized=${1^^}
    case "$normalized" in
        ""|0|OFF|NO|FALSE|N|IGNORE|NOTFOUND|*-NOTFOUND)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

verify_unfiltered_wheel_build() {
    local build_dir=$1
    local audit_log=$2
    local generated_file
    local found_filter=false
    local audited_files=0
    local cache_line
    local cache_entry
    local cache_value

    : >"$audit_log"
    while IFS= read -r -d '' generated_file; do
        audited_files=$((audited_files + 1))
        printf 'audited_file=%s\n' "$generated_file" >>"$audit_log"
        case "$(basename "$generated_file")" in
            custom_compile_options.ini)
                if grep -nE -- '--tiling[_-]key([=[:space:]]|$)' \
                    "$generated_file" >>"$audit_log"; then
                    found_filter=true
                fi
                ;;
            custom_tiling_keys.ini)
                if grep -n '[^[:space:]]' "$generated_file" >>"$audit_log"; then
                    found_filter=true
                fi
                ;;
            CMakeCache.txt)
                while IFS= read -r cache_line; do
                    cache_entry=${cache_line#*:}
                    cache_value=${cache_entry#*=}
                    if cmake_value_is_false "$cache_value"; then
                        printf 'tiling_key_cache_value=%s classification=disabled\n' \
                            "$cache_value" >>"$audit_log"
                    else
                        printf '%s\n' "$cache_line" >>"$audit_log"
                        found_filter=true
                    fi
                done < <(grep -nE -- '^TILING_KEY(:[^=]*)?=' \
                    "$generated_file" || true)
                ;;
        esac
    done < <(find "$build_dir" -type f \
        \( -name custom_compile_options.ini -o \
           -name custom_tiling_keys.ini -o \
           -name CMakeCache.txt \) -print0 2>/dev/null)

    if [[ $audited_files -eq 0 ]]; then
        echo "[FAIL] no generated wheel build configuration was available to audit" >&2
        return 1
    fi
    if $found_filter; then
        echo "[FAIL] generated wheel build configuration contains a tiling-key filter" >&2
        cat "$audit_log" >&2
        return 1
    fi
    echo "generated_wheel_tiling_filter=none audited_files=$audited_files" | \
        tee -a "$audit_log"
}

verify_runtime_identity() {
    [[ "$CURRENT_CANN_ENV" == "$BUILD_CANN_ENV" ]] || {
        echo "[FAIL] CANN env differs from build: $CURRENT_CANN_ENV != $BUILD_CANN_ENV" >&2
        exit 1
    }
    [[ "$CURRENT_CANN_ENV_SHA256" == "$BUILD_CANN_ENV_SHA256" ]] || {
        echo "[FAIL] selected CANN set_env.sh changed after build" >&2
        exit 1
    }
    [[ "$CURRENT_ASCEND_HOME" == "$BUILD_ASCEND_HOME" ]] || {
        echo "[FAIL] ASCEND_HOME_PATH differs from build" >&2
        exit 1
    }
    [[ "$CURRENT_CANN_VERSION_DIGEST" == "$BUILD_CANN_VERSION_DIGEST" ]] || {
        echo "[FAIL] selected CANN version files changed after build" >&2
        exit 1
    }
    [[ "$SOC" == "$BUILD_SOC" ]] || {
        echo "[FAIL] SoC differs from build: $SOC != $BUILD_SOC" >&2
        exit 1
    }
    [[ "$CURRENT_PYTHON" == "$BUILD_PYTHON" && \
       "$CURRENT_PYTHON_VERSION" == "$BUILD_PYTHON_VERSION" ]] || {
        echo "[FAIL] Python differs from build" >&2
        exit 1
    }
    [[ "$CURRENT_TORCH_VERSION" == "$BUILD_TORCH_VERSION" && \
       "$CURRENT_TORCH_NPU_VERSION" == "$BUILD_TORCH_NPU_VERSION" ]] || {
        echo "[FAIL] torch/torch_npu differs from build" >&2
        exit 1
    }
    [[ "$(sha256sum "$WHEEL" | awk '{print $1}')" == "$WHEEL_SHA256" ]] || {
        echo "[FAIL] wheel hash differs from build state" >&2
        exit 1
    }
    [[ "$(sha256sum "$VALIDATION_MANIFEST" | awk '{print $1}')" == \
       "$VALIDATION_MANIFEST_SHA256" ]] || {
        echo "[FAIL] validation-source manifest changed after build" >&2
        exit 1
    }
    (
        cd "$WHEEL_SRC"
        sha256sum -c "$VALIDATION_MANIFEST"
    ) | tee "$RUN_ROOT/logs/validation_sources.verify.log"
    local archived_runner current_runner_sha archived_runner_sha
    archived_runner="$WHEEL_SRC/scripts/run_chunk_kda_bwd_intra_grouped_validation.sh"
    [[ -f "$archived_runner" ]] || {
        echo "[FAIL] archived validation runner disappeared" >&2
        exit 1
    }
    current_runner_sha="$(sha256sum "$RUNNER_PATH" | awk '{print $1}')"
    archived_runner_sha="$(sha256sum "$archived_runner" | awk '{print $1}')"
    [[ "$current_runner_sha" == "$BUILD_RUNNER_SHA256" && \
       "$archived_runner_sha" == "$BUILD_RUNNER_SHA256" ]] || {
        echo "[FAIL] current runner differs from the clean-build runner" >&2
        echo "       current=$current_runner_sha archived=$archived_runner_sha" >&2
        echo "       expected=$BUILD_RUNNER_SHA256" >&2
        exit 1
    }
    verify_clean_catlass runtime
    local current_catlass current_catlass_tree
    current_catlass="$(git -C "$RUN_ROOT/third_party/catlass" rev-parse HEAD 2>/dev/null)" || {
        echo "[FAIL] pinned CATLASS checkout disappeared" >&2
        exit 1
    }
    current_catlass_tree="$(git -C "$RUN_ROOT/third_party/catlass" rev-parse 'HEAD^{tree}')"
    [[ "$current_catlass" == "$BUILD_CATLASS_COMMIT" && \
       "$current_catlass" == "$EXPECTED_CATLASS_COMMIT" && \
       "$current_catlass_tree" == "$BUILD_CATLASS_TREE" ]] || {
        echo "[FAIL] CATLASS revision differs from the clean build" >&2
        exit 1
    }
}

kda_object_digest() {
    local install_root=$1
    local -a objects=()
    mapfile -t objects < <(find "$install_root/fla_npu/opp" -type f \
        -path '*chunk_kda_bwd_intra*' -name '*.o' -print | sort)
    [[ ${#objects[@]} -gt 0 ]] || {
        echo "[FAIL] no KDA object under $install_root" >&2
        return 1
    }
    sha256sum "${objects[@]}" | awk '{print $1}' | sort | sha256sum | awk '{print $1}'
}

verify_runtime_install() {
    local install_root=$1 purpose=$2
    export VERIFY_INSTALL_ROOT="$install_root"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$install_root${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import inspect
import os
import pathlib

import fla_npu
from fla_npu.ops import ascendc

root = pathlib.Path(os.environ["VERIFY_INSTALL_ROOT"]).resolve()
module = pathlib.Path(inspect.getfile(fla_npu)).resolve()
assert root in module.parents, (root, module)
vendor = root / "fla_npu/opp/vendors/fla_npu_transformer"
assert list(vendor.rglob("chunk_kda_bwd_intra.json"))
assert list(vendor.rglob("chunk_kda_bwd_intra/*.o"))
assert callable(ascendc.chunk_kda_bwd_intra)
op_api = pathlib.Path(os.environ["FLA_NPU_OP_API_LIB"]).resolve()
assert root in op_api.parents, (root, op_api)
print("runtime module:", module)
print("runtime op_api:", op_api)
PY
    unset VERIFY_INSTALL_ROOT
    local digest
    digest="$(kda_object_digest "$install_root")"
    [[ "$digest" == "$WHEEL_KDA_DIGEST" ]] || {
        echo "[FAIL] $purpose KDA object digest differs from clean wheel" >&2
        exit 1
    }
    printf '%s\n' "$digest" >"$RUN_ROOT/logs/${purpose}_kda_object.digest"
}

create_runtime_install() {
    local purpose=$1 install_root
    install_root="$(mktemp -d "$RUN_ROOT/${purpose}_install.XXXXXX")"
    python3 -m pip install --no-deps --target "$install_root" "$WHEEL"
    verify_runtime_install "$install_root" "$purpose"
    RUNTIME_INSTALL="$install_root"
    RUNTIME_PYTHONPATH="$install_root${PYTHONPATH:+:$PYTHONPATH}"
    export RUNTIME_INSTALL RUNTIME_PYTHONPATH
}

probe_key() {
    local key=$1 src=$2
    local stage="$RUN_ROOT/probe${key}_opp"
    local log="$RUN_ROOT/logs/key${key}_compile.log"
    echo "===== clean compile probe: key $key ====="
    (
        cd "$src"
        bash build.sh --soc="$SOC" --pkg --vendor_name=fla_npu \
            --ops=chunk_kda_bwd_intra --tiling_key="$key" "-j$JOBS"
    ) 2>&1 | tee "$log"

    PROBE_SOURCE="$src" PROBE_KEY="$key" python3 - <<'PY'
import os
import pathlib
import re

root = pathlib.Path(os.environ["PROBE_SOURCE"]) / "build"
key = re.escape(os.environ["PROBE_KEY"])
pattern = re.compile(
    r"--tiling_key=(?:\"" + key + r"\"|'" + key + r"'|" + key + r")"
    r"(?:\s|\)|\\|$)"
)
matches = []
for path in root.rglob("*"):
    if not path.is_file() or path.suffix not in {".sh", ".txt", ".log", ".cmake", ".json"}:
        continue
    if path.stat().st_size > 2 * 1024 * 1024:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    if pattern.search(text):
        matches.append(path)
if not matches:
    raise SystemExit(f"generated build commands do not prove exact tiling key {os.environ['PROBE_KEY']}")
print("exact key filter:", *matches, sep="\n  ")
PY
    local -a runs=()
    mapfile -t runs < <(find "$src/build_out" -maxdepth 1 -type f \
        -name 'fla-npu-*.run' -print | sort)
    [[ ${#runs[@]} -eq 1 ]] || {
        echo "[FAIL] key$key run package count=${#runs[@]}" >&2
        exit 1
    }
    mkdir -p "$stage"
    chmod +x "${runs[0]}"
    "${runs[0]}" --quiet --install-path="$stage"
    local -a objects=()
    mapfile -t objects < <(find "$stage" -type f \
        -path '*chunk_kda_bwd_intra*' -name '*.o' -print | sort)
    [[ ${#objects[@]} -gt 0 ]] || { echo "[FAIL] key$key has no KDA object" >&2; exit 1; }
    sha256sum "${objects[@]}" | tee "$RUN_ROOT/logs/key${key}_objects.sha256"
}

read_bool_constant() {
    local file=$1 symbol=$2
    python3 - "$file" "$symbol" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
symbol = sys.argv[2]
text = path.read_text(encoding="utf-8")
pattern = re.compile(
    rf"^\s*constexpr\s+bool\s+{re.escape(symbol)}\s*=\s*(true|false)\s*;\s*$",
    re.MULTILINE,
)
matches = pattern.findall(text)
if len(matches) != 1:
    raise SystemExit(f"expected one {symbol} definition in {path}, found {len(matches)}")
print(matches[0])
PY
}

rewrite_bool_constant() {
    local file=$1 symbol=$2 value=$3
    python3 - "$file" "$symbol" "$value" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
symbol = sys.argv[2]
value = sys.argv[3]
if value not in {"true", "false"}:
    raise SystemExit(f"invalid bool literal: {value}")
text = path.read_text(encoding="utf-8")
pattern = re.compile(
    rf"^(\s*constexpr\s+bool\s+{re.escape(symbol)}\s*=\s*)(true|false)(\s*;\s*)$",
    re.MULTILINE,
)
updated, count = pattern.subn(rf"\g<1>{value}\g<3>", text)
if count != 1:
    raise SystemExit(f"expected one {symbol} definition in {path}, found {count}")
path.write_text(updated, encoding="utf-8")
PY
}

read_uint_constant() {
    local file=$1 symbol=$2
    python3 - "$file" "$symbol" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
symbol = sys.argv[2]
text = path.read_text(encoding="utf-8")
pattern = re.compile(
    rf"^\s*constexpr\s+uint32_t\s+{re.escape(symbol)}\s*=\s*([0-9]+)\s*;\s*$",
    re.MULTILINE,
)
matches = pattern.findall(text)
if len(matches) != 1:
    raise SystemExit(f"expected one {symbol} definition in {path}, found {len(matches)}")
print(matches[0])
PY
}

rewrite_uint_constant() {
    local file=$1 symbol=$2 value=$3
    [[ "$value" =~ ^[0-9]+$ ]] || {
        echo "[FAIL] invalid uint literal for $symbol: $value" >&2
        exit 2
    }
    python3 - "$file" "$symbol" "$value" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
symbol = sys.argv[2]
value = sys.argv[3]
text = path.read_text(encoding="utf-8")
pattern = re.compile(
    rf"^(\s*constexpr\s+uint32_t\s+{re.escape(symbol)}\s*=\s*)([0-9]+)(\s*;\s*)$",
    re.MULTILINE,
)
updated, count = pattern.subn(rf"\g<1>{value}\g<3>", text)
if count != 1:
    raise SystemExit(f"expected one {symbol} definition in {path}, found {count}")
path.write_text(updated, encoding="utf-8")
PY
}

resolve_build_variant() {
    local header="$SOURCE_ROOT/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra_grouped.hpp"
    local factor_value overlap_value epilogue_value scratch_value
    local tail_blocks_value task_store_value persistent_mmad_value vector_mask_value db_reduce_value stage_a_value cube_mode_value stage_io_value aic_diagnostic_value
    factor_value="$(read_bool_constant "$header" KDA_GROUPED_FACTOR_PAIR_GATES)"
    overlap_value="$(read_bool_constant "$header" KDA_GROUPED_OVERLAP_SHARED_SETUP)"
    epilogue_value="$(read_bool_constant "$header" KDA_GROUPED_OVERLAP_STAGE_EPILOGUE)"
    scratch_value="$(read_bool_constant "$header" KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH)"
    tail_blocks_value="$(read_bool_constant "$header" KDA_GROUPED_BATCH_TAIL_BLOCKS)"
    task_store_value="$(read_bool_constant "$header" KDA_GROUPED_OVERLAP_TASK_STORE)"
    persistent_mmad_value="$(read_bool_constant "$header" KDA_GROUPED_PERSISTENT_MMAD_ENGINES)"
    vector_mask_value="$(read_bool_constant "$header" KDA_GROUPED_REUSE_VECTOR_MASK)"
    db_reduce_value="$(read_bool_constant "$header" KDA_GROUPED_COALESCE_DB_REDUCE)"
    stage_a_value="$(read_bool_constant "$header" KDA_GROUPED_PACK_STAGE_A)"
    cube_mode_value="$(read_bool_constant "$header" KDA_GROUPED_USE_HF32_CUBE)"
    stage_io_value="$(read_bool_constant "$header" KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER)"
    aic_diagnostic_value="$(read_uint_constant "$header" KDA_GROUPED_AIC_DIAGNOSTIC_MODE)"
    case "$PAIR_GATES_MODE" in
        source) ;;
        factor) factor_value=true ;;
        direct) factor_value=false ;;
    esac
    case "$SHARED_SETUP_MODE" in
        source) ;;
        overlap) overlap_value=true ;;
        prologue) overlap_value=false ;;
    esac
    case "$STAGE_EPILOGUE_MODE" in
        source) ;;
        overlap) epilogue_value=true ;;
        tail) epilogue_value=false ;;
    esac
    case "$PAIR_SCRATCH_MODE" in
        source) ;;
        pingpong) scratch_value=true ;;
        single) scratch_value=false ;;
    esac
    case "$TAIL_BLOCKS_MODE" in
        source) ;;
        batch) tail_blocks_value=true ;;
        scalar) tail_blocks_value=false ;;
    esac
    case "$TASK_STORE_MODE" in
        source) ;;
        overlap) task_store_value=true ;;
        serial) task_store_value=false ;;
    esac
    case "$MMAD_ENGINES_MODE" in
        source) ;;
        persistent) persistent_mmad_value=true ;;
        scoped) persistent_mmad_value=false ;;
    esac
    case "$VECTOR_MASK_MODE" in
        source) ;;
        reuse) vector_mask_value=true ;;
        per-call) vector_mask_value=false ;;
    esac
    case "$DB_REDUCE_MODE" in
        source) ;;
        coalesced) db_reduce_value=true ;;
        per-row) db_reduce_value=false ;;
    esac
    case "$STAGE_A_MODE" in
        source) ;;
        packed) stage_a_value=true ;;
        split) stage_a_value=false ;;
    esac
    case "$CUBE_MODE" in
        source) ;;
        hf32) cube_mode_value=true ;;
        ieee) cube_mode_value=false ;;
    esac
    case "$STAGE_IO_MODE" in
        source) ;;
        tscm) stage_io_value=true ;;
        gm) stage_io_value=false ;;
    esac
    case "$AIC_DIAGNOSTIC_MODE" in
        source) ;;
        full) aic_diagnostic_value=0 ;;
        handshake) aic_diagnostic_value=1 ;;
        stage0-right) aic_diagnostic_value=2 ;;
        stage0-left) aic_diagnostic_value=3 ;;
        stage0-both) aic_diagnostic_value=4 ;;
        through-stage1) aic_diagnostic_value=5 ;;
        through-stage2) aic_diagnostic_value=6 ;;
    esac
    [[ "$factor_value" == true ]] && BUILD_PAIR_GATES=factor || BUILD_PAIR_GATES=direct
    [[ "$overlap_value" == true ]] && BUILD_SHARED_SETUP=overlap || BUILD_SHARED_SETUP=prologue
    [[ "$epilogue_value" == true ]] && BUILD_STAGE_EPILOGUE=overlap || BUILD_STAGE_EPILOGUE=tail
    [[ "$scratch_value" == true ]] && BUILD_PAIR_SCRATCH=pingpong || BUILD_PAIR_SCRATCH=single
    [[ "$tail_blocks_value" == true ]] && BUILD_TAIL_BLOCKS=batch || BUILD_TAIL_BLOCKS=scalar
    [[ "$task_store_value" == true ]] && BUILD_TASK_STORE=overlap || BUILD_TASK_STORE=serial
    [[ "$persistent_mmad_value" == true ]] && BUILD_MMAD_ENGINES=persistent || BUILD_MMAD_ENGINES=scoped
    [[ "$vector_mask_value" == true ]] && BUILD_VECTOR_MASK=reuse || BUILD_VECTOR_MASK=per-call
    [[ "$db_reduce_value" == true ]] && BUILD_DB_REDUCE=coalesced || BUILD_DB_REDUCE=per-row
    [[ "$stage_a_value" == true ]] && BUILD_STAGE_A=packed || BUILD_STAGE_A=split
    [[ "$cube_mode_value" == true ]] && BUILD_CUBE_MODE=hf32 || BUILD_CUBE_MODE=ieee
    [[ "$stage_io_value" == true ]] && BUILD_STAGE_IO=tscm || BUILD_STAGE_IO=gm
    case "$aic_diagnostic_value" in
        0) BUILD_AIC_DIAGNOSTIC=full ;;
        1) BUILD_AIC_DIAGNOSTIC=handshake ;;
        2) BUILD_AIC_DIAGNOSTIC=stage0-right ;;
        3) BUILD_AIC_DIAGNOSTIC=stage0-left ;;
        4) BUILD_AIC_DIAGNOSTIC=stage0-both ;;
        5) BUILD_AIC_DIAGNOSTIC=through-stage1 ;;
        6) BUILD_AIC_DIAGNOSTIC=through-stage2 ;;
        *) echo "[FAIL] invalid KDA_GROUPED_AIC_DIAGNOSTIC_MODE=$aic_diagnostic_value" >&2; exit 2 ;;
    esac
    if [[ "$BUILD_STAGE_IO" == tscm && "$SOC" == ascend910b ]]; then
        echo "[FAIL] A2/ascend910b UB->TSCM is software-emulated through GM/Matmul KFC;" >&2
        echo "       this direct CATLASS kernel must use --stage-io gm" >&2
        exit 2
    fi
    if [[ "$BUILD_TASK_STORE" == overlap ]]; then
        [[ "$BUILD_TAIL_BLOCKS" == batch && "$BUILD_STAGE_EPILOGUE" == tail ]] || {
            echo "[FAIL] task-store overlap requires --tail-blocks batch --stage-epilogue tail" >&2
            exit 2
        }
    fi
    BUILD_VARIANT_ID="pair-${BUILD_PAIR_GATES}_setup-${BUILD_SHARED_SETUP}_epilogue-${BUILD_STAGE_EPILOGUE}_scratch-${BUILD_PAIR_SCRATCH}_tail-${BUILD_TAIL_BLOCKS}_mmad-${BUILD_MMAD_ENGINES}_vmask-${BUILD_VECTOR_MASK}_dbr-${BUILD_DB_REDUCE}_store-${BUILD_TASK_STORE}_stagea-${BUILD_STAGE_A}_cube-${BUILD_CUBE_MODE}_io-${BUILD_STAGE_IO}"
    if [[ "$BUILD_AIC_DIAGNOSTIC" != full ]]; then
        BUILD_VARIANT_ID+="_aicdiag-${BUILD_AIC_DIAGNOSTIC}"
    fi
}

apply_build_variant() {
    local root=$1
    local header="$root/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra_grouped.hpp"
    local factor_value=false overlap_value=false epilogue_value=false
    local scratch_value=false tail_blocks_value=false task_store_value=false persistent_mmad_value=false
    local vector_mask_value=false db_reduce_value=false stage_a_value=false cube_mode_value=false stage_io_value=false
    local aic_diagnostic_value=0
    [[ "$BUILD_PAIR_GATES" == factor ]] && factor_value=true
    [[ "$BUILD_SHARED_SETUP" == overlap ]] && overlap_value=true
    [[ "$BUILD_STAGE_EPILOGUE" == overlap ]] && epilogue_value=true
    [[ "$BUILD_PAIR_SCRATCH" == pingpong ]] && scratch_value=true
    [[ "$BUILD_TAIL_BLOCKS" == batch ]] && tail_blocks_value=true
    [[ "$BUILD_TASK_STORE" == overlap ]] && task_store_value=true
    [[ "$BUILD_MMAD_ENGINES" == persistent ]] && persistent_mmad_value=true
    [[ "$BUILD_VECTOR_MASK" == reuse ]] && vector_mask_value=true
    [[ "$BUILD_DB_REDUCE" == coalesced ]] && db_reduce_value=true
    [[ "$BUILD_STAGE_A" == packed ]] && stage_a_value=true
    [[ "$BUILD_CUBE_MODE" == hf32 ]] && cube_mode_value=true
    [[ "$BUILD_STAGE_IO" == tscm ]] && stage_io_value=true
    case "$BUILD_AIC_DIAGNOSTIC" in
        full) aic_diagnostic_value=0 ;;
        handshake) aic_diagnostic_value=1 ;;
        stage0-right) aic_diagnostic_value=2 ;;
        stage0-left) aic_diagnostic_value=3 ;;
        stage0-both) aic_diagnostic_value=4 ;;
        through-stage1) aic_diagnostic_value=5 ;;
        through-stage2) aic_diagnostic_value=6 ;;
        *) echo "[FAIL] invalid build AIC diagnostic mode: $BUILD_AIC_DIAGNOSTIC" >&2; exit 2 ;;
    esac
    rewrite_bool_constant "$header" KDA_GROUPED_FACTOR_PAIR_GATES "$factor_value"
    rewrite_bool_constant "$header" KDA_GROUPED_OVERLAP_SHARED_SETUP "$overlap_value"
    rewrite_bool_constant "$header" KDA_GROUPED_OVERLAP_STAGE_EPILOGUE "$epilogue_value"
    rewrite_bool_constant "$header" KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH "$scratch_value"
    rewrite_bool_constant "$header" KDA_GROUPED_BATCH_TAIL_BLOCKS "$tail_blocks_value"
    rewrite_bool_constant "$header" KDA_GROUPED_OVERLAP_TASK_STORE "$task_store_value"
    rewrite_bool_constant "$header" KDA_GROUPED_PERSISTENT_MMAD_ENGINES "$persistent_mmad_value"
    rewrite_bool_constant "$header" KDA_GROUPED_REUSE_VECTOR_MASK "$vector_mask_value"
    rewrite_bool_constant "$header" KDA_GROUPED_COALESCE_DB_REDUCE "$db_reduce_value"
    rewrite_bool_constant "$header" KDA_GROUPED_PACK_STAGE_A "$stage_a_value"
    rewrite_bool_constant "$header" KDA_GROUPED_USE_HF32_CUBE "$cube_mode_value"
    rewrite_bool_constant "$header" KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER "$stage_io_value"
    rewrite_uint_constant "$header" KDA_GROUPED_AIC_DIAGNOSTIC_MODE "$aic_diagnostic_value"
    [[ "$(read_bool_constant "$header" KDA_GROUPED_FACTOR_PAIR_GATES)" == "$factor_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_OVERLAP_SHARED_SETUP)" == "$overlap_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_OVERLAP_STAGE_EPILOGUE)" == "$epilogue_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH)" == "$scratch_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_BATCH_TAIL_BLOCKS)" == "$tail_blocks_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_OVERLAP_TASK_STORE)" == "$task_store_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_PERSISTENT_MMAD_ENGINES)" == "$persistent_mmad_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_REUSE_VECTOR_MASK)" == "$vector_mask_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_COALESCE_DB_REDUCE)" == "$db_reduce_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_PACK_STAGE_A)" == "$stage_a_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_USE_HF32_CUBE)" == "$cube_mode_value" ]]
    [[ "$(read_bool_constant "$header" KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER)" == "$stage_io_value" ]]
    [[ "$(read_uint_constant "$header" KDA_GROUPED_AIC_DIAGNOSTIC_MODE)" == "$aic_diagnostic_value" ]]
}

run_build() {
    source_cann
    cd "$SOURCE_ROOT"
    git rev-parse --is-inside-work-tree >/dev/null
    git diff --quiet || { echo "[FAIL] tracked worktree changes must be committed" >&2; exit 1; }
    git diff --cached --quiet || { echo "[FAIL] staged changes must be committed" >&2; exit 1; }
    COMMIT="$(git rev-parse HEAD)"
    resolve_build_variant
    BUILD_CANN_ENV="$CURRENT_CANN_ENV"
    BUILD_CANN_ENV_SHA256="$CURRENT_CANN_ENV_SHA256"
    BUILD_ASCEND_HOME="$CURRENT_ASCEND_HOME"
    BUILD_CANN_VERSION_DIGEST="$CURRENT_CANN_VERSION_DIGEST"
    BUILD_SOC="$SOC"
    BUILD_PYTHON="$CURRENT_PYTHON"
    BUILD_PYTHON_VERSION="$CURRENT_PYTHON_VERSION"
    BUILD_TORCH_VERSION="$CURRENT_TORCH_VERSION"
    BUILD_TORCH_NPU_VERSION="$CURRENT_TORCH_NPU_VERSION"
    local required
    for required in "${VALIDATION_SOURCES[@]}"; do
        git cat-file -e "$COMMIT:$required" || {
            echo "[FAIL] required file is not committed: $required" >&2
            exit 1
        }
    done

    local stamp run_base_real shared_3p probe23 probe15
    stamp="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_BASE"
    run_base_real="$(cd "$RUN_BASE" && pwd -P)"
    RUN_ROOT="$(mktemp -d \
        "$run_base_real/wys_kda_grouped_${stamp}_${BUILD_VARIANT_ID}_XXXXXX")"
    probe23="$RUN_ROOT/probe23"
    probe15="$RUN_ROOT/probe15"
    WHEEL_SRC="$RUN_ROOT/wheel_src"
    shared_3p="$RUN_ROOT/third_party"
    mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/artifacts" "$RUN_ROOT/tmp" \
        "$shared_3p" "$probe23" "$probe15" "$WHEEL_SRC"
    export TMPDIR="$RUN_ROOT/tmp"

    local dst
    for dst in "$probe23" "$probe15" "$WHEEL_SRC"; do
        git archive --format=tar "$COMMIT" | tar -xf - -C "$dst"
        [[ ! -e "$dst/third_party" ]] || {
            echo "[FAIL] archive unexpectedly contains third_party: $dst" >&2
            exit 1
        }
        ln -s "$shared_3p" "$dst/third_party"
        apply_build_variant "$dst"
    done
    local grouped_header_rel="fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra_grouped.hpp"
    local probe23_header_sha probe15_header_sha wheel_header_sha
    probe23_header_sha="$(sha256sum "$probe23/$grouped_header_rel" | awk '{print $1}')"
    probe15_header_sha="$(sha256sum "$probe15/$grouped_header_rel" | awk '{print $1}')"
    wheel_header_sha="$(sha256sum "$WHEEL_SRC/$grouped_header_rel" | awk '{print $1}')"
    [[ "$probe23_header_sha" == "$probe15_header_sha" && \
       "$probe23_header_sha" == "$wheel_header_sha" ]] || {
        echo "[FAIL] A/B variant rewrite differs across clean build sources" >&2
        exit 1
    }
    VALIDATION_MANIFEST="$RUN_ROOT/validation_sources.sha256"
    (
        cd "$WHEEL_SRC"
        sha256sum "${VALIDATION_SOURCES[@]}"
    ) >"$VALIDATION_MANIFEST"
    VALIDATION_MANIFEST_SHA256="$(sha256sum "$VALIDATION_MANIFEST" | awk '{print $1}')"
    BUILD_RUNNER_SHA256="$(sha256sum \
        "$WHEEL_SRC/scripts/run_chunk_kda_bwd_intra_grouped_validation.sh" | awk '{print $1}')"
    [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$BUILD_RUNNER_SHA256" ]] || {
        echo "[FAIL] invoked runner differs from the committed clean-build runner" >&2
        exit 1
    }
    (
        cd "$WHEEL_SRC"
        sha256sum --check --quiet --strict "$VALIDATION_MANIFEST"
    )
    python3 "$WHEEL_SRC/scripts/model_chunk_kda_bwd_intra_blockdiag.py" \
        --json >"$RUN_ROOT/logs/blockdiag_model.json"
    python3 - "$RUN_ROOT/logs/blockdiag_model.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["calls"] == {
    "logical_per_task": 17,
    "fused_per_task": 4,
    "reduction": 4.25,
}
assert report["compute"]["inflation"] == 5.45
assert report["precision_contract"]["gate_reference_reassociation"] is False
assert report["precision_contract"]["active_local_k_order_preserved"] is True

candidate = report["layout_feasible_candidate"]
assert candidate["calls"] == 11
assert candidate["fmas_per_task"] == 2_424_832
assert candidate["workspace_traffic_bytes_per_task"] == 1_146_880
assert candidate["routing_proof"] == "PASS"
assert candidate["precision_contract"]["gate_reference_reassociation"] is False
assert candidate["precision_contract"]["active_product_order_changed"] is False
print("[PASS] block-diagonal design model")
PY
    python3 "$WHEEL_SRC/scripts/model_chunk_kda_bwd_intra_traffic.py" \
        --stage-io "$BUILD_STAGE_IO" --pair-scratch "$BUILD_PAIR_SCRATCH" --json \
        >"$RUN_ROOT/logs/traffic_model.json"
    python3 - "$RUN_ROOT/logs/traffic_model.json" "$BUILD_PAIR_SCRATCH" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected_pair_scratch = sys.argv[2]
assert report["target"]["tasks"] == 4096
assert report["workspace"]["queue_depth"] == 2
assert report["workspace"]["total_bytes_per_task"] == 946_176
assert report["workspace"]["total_bytes_target"] == 3_875_536_896
assert report["tensors"]["total_bytes_per_task"] == 293_632
assert report["total"]["bytes_target"] == 5_078_253_568
assert report["scope"]["workspace_slots_reduce_traffic"] is False
assert report["implementation"]["pair_scratch"] == expected_pair_scratch
assert report["on_chip_capacity_only"]["full_ub_double_fits"] is False
assert report["scenarios"]["current_gm_bridge"][
    "required_gbps_at_target_ms"
] == 1269.563392
assert report["scenarios"]["ab_on_chip"]["bytes_target"] == 2_813_329_408
a_reuse = report["aic_l1_a_reuse_candidate"]
assert a_reuse["implemented"] is False
assert a_reuse["saved_bytes_per_task"] == 20_480
assert a_reuse["saved_bytes_target"] == 83_886_080
assert report["scenarios"]["aic_l1_a_reuse"]["bytes_target"] == 4_994_367_488
expected_active = "current_gm_bridge"
assert report["implementation"]["active_scenario"] == expected_active
assert report["implementation"]["a2_physical_direct_ub_to_l1"] is False
off_right = report["off_right_consume_coalescing"]
assert off_right["changes_fp32_arithmetic"] is False
assert off_right["changes_gm_bytes"] is False
assert off_right["bytes_target"] == 201_326_592
assert off_right["mte2_calls_target_before"] == 49_152
assert off_right["mte2_calls_target_after"] == 24_576
assert off_right["mul_add_calls_target_saved"] == 24_576
right_b = report["right_b_write_coalescing"]
assert right_b["changes_fp32_arithmetic"] is False
assert right_b["changes_gm_bytes"] is False
assert right_b["bytes_target"] == 671_088_640
assert right_b["mte3_calls_target_before"] == 163_840
assert right_b["mte3_calls_target_after"] == 81_920
left_c = report["left_c_read_coalescing"]
assert left_c["changes_fp32_arithmetic"] is False
assert left_c["changes_gm_bytes"] is False
assert left_c["bytes_target"] == 469_762_048
assert left_c["mte2_calls_target_before"] == 114_688
assert left_c["mte2_calls_target_after"] == 57_344
batch_tail = report["batch_tail_coalescing"]
assert batch_tail["changes_per_element_fp32_order"] is False
assert batch_tail["changes_gm_bytes"] is False
assert batch_tail["db_input_bytes_target"] == 1_048_576
assert batch_tail["feature_input_bytes_target"] == 402_653_184
assert batch_tail["output_bytes_target"] == 403_701_760
assert batch_tail["total_bytes_target"] == 807_403_520
assert batch_tail["total_mte2_calls_target_before"] == 131_072
assert batch_tail["total_mte2_calls_target_after"] == 32_768
assert batch_tail["vector_expression_calls_target_before"] == 196_608
assert batch_tail["vector_expression_calls_target_after"] == 49_152
assert batch_tail["mte3_calls_target_before"] == 131_072
assert batch_tail["mte3_calls_target_after"] == 32_768
scalar_hot = report["scalar_hot_path_hoisting"]
assert scalar_hot["source_level_model"] is True
assert scalar_hot["changes_fp32_arithmetic"] is False
assert scalar_hot["changes_vector_instructions"] is False
assert scalar_hot["changes_gm_bytes"] is False
assert scalar_hot["task_aiv_pairs_target"] == 8_192
assert scalar_hot["row_start_source_expressions_target_before"] == 114_688
assert scalar_hot["row_start_source_expressions_target_after"] == 0
assert scalar_hot["consume_get_subblock_reads_target_before"] == 32_768
assert scalar_hot["consume_get_subblock_reads_target_after"] == 0
assert scalar_hot["persistent_gate_loop_iterations_target_before"] == 24_576
assert scalar_hot["persistent_gate_loop_iterations_target_after"] == 0
assert scalar_hot["kernel_entry_get_subblock_reads_unchanged"] is True
persistent_mmad = report["persistent_mmad_scheduling"]
assert persistent_mmad["source_default_enabled"] is True
assert persistent_mmad["changes_fp32_arithmetic"] is False
assert persistent_mmad["changes_logical_gemm_order"] is False
assert persistent_mmad["changes_gm_bytes"] is False
assert persistent_mmad["logical_gemm_calls_target"] == 69_632
assert persistent_mmad["scoped_envelopes_target"] == 69_632
assert persistent_mmad["persistent_engines_per_aic"] == 2
assert persistent_mmad["persistent_event_owners_per_aic"] == 1
assert persistent_mmad["persistent_envelopes_target"] == 20
assert persistent_mmad["scoped_one_envelope_per_logical_gemm"] is True
assert persistent_mmad["shared_event_domain_serializes_both_layouts"] is True
assert persistent_mmad["scoped_compile_time_rollback_retained"] is True
pair_scratch = report["pair_scratch_pingpong"]
assert pair_scratch["source_default_enabled"] is False
assert pair_scratch["active"] is (expected_pair_scratch == "pingpong")
assert pair_scratch["changes_fp32_arithmetic"] is False
assert pair_scratch["changes_gm_bytes"] is False
assert pair_scratch["changes_workspace_layout"] is False
assert pair_scratch["pairs_per_task"] == 6
assert pair_scratch["mte3_bytes_per_task"] == 147_456
assert pair_scratch["mte3_bytes_target_eligible_for_overlap"] == 603_979_776
assert pair_scratch["pair_completion_waits_target_before"] == 49_152
assert pair_scratch["pair_completion_waits_target_after"] == 24_576
assert pair_scratch["pair_completion_waits_target_saved"] == 24_576
assert pair_scratch["single_ub_bytes"] == 179_936
assert pair_scratch["pingpong_ub_bytes"] == 192_256
assert pair_scratch["active_ub_bytes"] == (
    192_256 if expected_pair_scratch == "pingpong" else 179_936
)
assert pair_scratch["extra_ub_bytes"] == 12_320
assert pair_scratch["a2_ub_headroom_bytes"] == 4_352
assert pair_scratch["baseline_live_mte3_v_event_ids"] == 1
assert pair_scratch["pingpong_live_mte3_v_event_ids"] == 3
assert pair_scratch["a2_mte3_v_event_id_capacity"] == 8
assert pair_scratch["requires_device_profile_to_claim_hidden_bytes"] is True
cv_paths = report["a2_cv_direct_paths"]
assert cv_paths["target_arch"] == "DAV_2201"
assert cv_paths["l0c_to_aiv_ub_supported"] is False
assert cv_paths["aiv_ub_to_l1_supported"] is False
assert cv_paths["direct_paths_start_at"] == "DAV_3510"
assert cv_paths["c_workspace_round_trip_still_required"] is True
tscm = report["tscm_ab_double_buffer"]
assert tscm["slot_bytes"] == 106_496
assert tscm["total_bytes"] == 212_992
assert tscm["total_plus_persistent_mmad_bytes"] == 290_816
assert tscm["l1_headroom_bytes"] == 233_472
assert tscm["fits_l1"] is True
assert tscm["a2_supported_direct_path"] is False
assert tscm["physical_layout"]["validated"] is True
assert tscm["physical_layout"]["same_physical_image_for_transpose"] is True
assert tscm["physical_layout"]["local_nd2nz_required"] is False
print("[PASS] grouped physical GM traffic model")
PY
    printf 'commit=%s\nsource=%s\ncann_env=%s\nvariant=%s\npair_gates=%s\nshared_setup=%s\nstage_epilogue=%s\npair_scratch=%s\ntail_blocks=%s\ntask_store=%s\nmmad_engines=%s\nvector_mask=%s\ndb_reduce=%s\nstage_a=%s\ncube_mode=%s\nstage_io=%s\naic_diagnostic=%s\ngrouped_header_sha256=%s\n' \
        "$COMMIT" "$SOURCE_ROOT" "$CANN_ENV" "$BUILD_VARIANT_ID" \
        "$BUILD_PAIR_GATES" "$BUILD_SHARED_SETUP" "$BUILD_STAGE_EPILOGUE" \
        "$BUILD_PAIR_SCRATCH" "$BUILD_TAIL_BLOCKS" "$BUILD_TASK_STORE" "$BUILD_MMAD_ENGINES" \
        "$BUILD_VECTOR_MASK" "$BUILD_DB_REDUCE" "$BUILD_STAGE_A" "$BUILD_CUBE_MODE" \
        "$BUILD_STAGE_IO" "$BUILD_AIC_DIAGNOSTIC" \
        "$wheel_header_sha" | \
        tee "$RUN_ROOT/build_identity.txt"
    python3 "$WHEEL_SRC/scripts/check_npu_env.py" --build-only
    command -v msprof || true

    probe_key 23 "$probe23"
    verify_clean_catlass key23
    BUILD_CATLASS_COMMIT="$(git -C "$shared_3p/catlass" rev-parse HEAD 2>/dev/null)" || {
        echo "[FAIL] key23 build did not produce a verifiable CATLASS checkout" >&2
        exit 1
    }
    BUILD_CATLASS_TREE="$(git -C "$shared_3p/catlass" rev-parse 'HEAD^{tree}')"
    [[ "$BUILD_CATLASS_COMMIT" == "$EXPECTED_CATLASS_COMMIT" ]] || {
        echo "[FAIL] unexpected CATLASS revision: $BUILD_CATLASS_COMMIT" >&2
        exit 1
    }
    printf 'catlass_commit=%s\ncatlass_tree=%s\nvalidation_manifest_sha256=%s\nrunner_sha256=%s\n' \
        "$BUILD_CATLASS_COMMIT" "$BUILD_CATLASS_TREE" \
        "$VALIDATION_MANIFEST_SHA256" "$BUILD_RUNNER_SHA256" | \
        tee -a "$RUN_ROOT/build_identity.txt"
    probe_key 15 "$probe15"

    echo "===== clean, unfiltered single-op wheel ====="
    cd "$WHEEL_SRC"
    unset TILING_KEY
    export FLA_NPU_OPS=chunk_kda_bwd_intra
    python3 -m pip wheel -v --no-build-isolation --no-deps . \
        -w "$RUN_ROOT/artifacts" 2>&1 | tee "$RUN_ROOT/logs/wheel_build.log"
    grep -Fq -- '--ops=chunk_kda_bwd_intra' "$RUN_ROOT/logs/wheel_build.log"
    if grep -F 'START bash build.sh' "$RUN_ROOT/logs/wheel_build.log" | \
        grep -q -- '--tiling_key'; then
        echo "[FAIL] validation wheel accidentally inherited a tiling-key filter" >&2
        exit 1
    fi
    verify_unfiltered_wheel_build "$WHEEL_SRC/build" \
        "$RUN_ROOT/logs/wheel_tiling_filter_audit.log"
    grep -Fq 'TILING_KEY_IS(15)' \
        "$WHEEL_SRC/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra.cpp"
    grep -Fq 'TILING_KEY_IS(23)' \
        "$WHEEL_SRC/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra.cpp"
    local -a wheels=()
    mapfile -t wheels < <(find "$RUN_ROOT/artifacts" -maxdepth 1 -type f \
        -name 'flash_linear_attention_npu-*.whl' -print | sort)
    [[ ${#wheels[@]} -eq 1 ]] || { echo "[FAIL] wheel count=${#wheels[@]}" >&2; exit 1; }
    WHEEL="${wheels[0]}"
    WHEEL_INSTALL="$RUN_ROOT/wheel_install"
    mkdir -p "$WHEEL_INSTALL"
    python3 -m pip install --no-deps --target "$WHEEL_INSTALL" "$WHEEL"
    sha256sum "$WHEEL" | tee "$RUN_ROOT/logs/wheel.sha256"
    WHEEL_SHA256="$(sha256sum "$WHEEL" | awk '{print $1}')"

    export WHEEL_INSTALL
    local runtime_pythonpath="$WHEEL_INSTALL${PYTHONPATH:+:$PYTHONPATH}"
    PYTHONPATH="$runtime_pythonpath" python3 - <<'PY'
import inspect
import os
import pathlib

import fla_npu
from fla_npu.ops import ascendc

root = pathlib.Path(os.environ["WHEEL_INSTALL"]).resolve()
module = pathlib.Path(inspect.getfile(fla_npu)).resolve()
assert root in module.parents, (root, module)
vendor = root / "fla_npu/opp/vendors/fla_npu_transformer"
configs = list(vendor.rglob("chunk_kda_bwd_intra.json"))
objects = list(vendor.rglob("chunk_kda_bwd_intra/*.o"))
assert callable(ascendc.chunk_kda_bwd_intra)
assert configs, "missing chunk_kda_bwd_intra.json"
assert objects, "missing KDA device object"
op_api = pathlib.Path(os.environ["FLA_NPU_OP_API_LIB"]).resolve()
assert root in op_api.parents, op_api
assert str(root / "fla_npu/opp") in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
print("fla_npu:", module)
print("op_api:", op_api)
print("KDA configs:", *configs, sep="\n  ")
print("KDA objects:", *objects, sep="\n  ")
PY
    find "$WHEEL_INSTALL/fla_npu/opp" -type f \
        -path '*chunk_kda_bwd_intra*' -name '*.o' -print0 | sort -z | \
        xargs -0 sha256sum | tee "$RUN_ROOT/logs/wheel_kda_objects.sha256"
    WHEEL_KDA_DIGEST="$(kda_object_digest "$WHEEL_INSTALL")"
    printf '%s\n' "$WHEEL_KDA_DIGEST" >"$RUN_ROOT/logs/wheel_kda_object.digest"

    verify_clean_catlass final-wheel
    [[ "$(git -C "$shared_3p/catlass" rev-parse HEAD)" == "$BUILD_CATLASS_COMMIT" && \
       "$(git -C "$shared_3p/catlass" rev-parse 'HEAD^{tree}')" == \
           "$BUILD_CATLASS_TREE" ]] || {
        echo "[FAIL] CATLASS changed while building the validation wheel" >&2
        exit 1
    }

    STATE_FILE="$RUN_ROOT/state.env"
    write_state "$STATE_FILE"
    echo "[PASS] build state: $STATE_FILE"
}

runtime_env() {
    source_cann
    verify_runtime_identity
}

run_test() {
    [[ -n "${RUN_ROOT:-}" ]] || load_state
    runtime_env
    rm -f -- "$RUN_ROOT/full37.pass" "$RUN_ROOT/full37_wheel.sha256"
    create_runtime_install test
    python3 -c 'import pytest; print("pytest:", pytest.__version__)'
    local test_dir="$WHEEL_SRC/torch_custom/fla_npu/test"
    local test_file="test_npu_chunk_kda_bwd_intra.py"
    cd "$test_dir"

    timeout --kill-after=15s 180s env \
        ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" TEST_DEVICE_ID=0 \
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
        KDA_EXPECT_PAIR_GATES="$BUILD_PAIR_GATES" \
        KDA_EXPECT_SHARED_SETUP="$BUILD_SHARED_SETUP" \
        KDA_EXPECT_STAGE_EPILOGUE="$BUILD_STAGE_EPILOGUE" \
        KDA_EXPECT_PAIR_SCRATCH="$BUILD_PAIR_SCRATCH" \
        KDA_EXPECT_TAIL_BLOCKS="$BUILD_TAIL_BLOCKS" \
        KDA_EXPECT_TASK_STORE="$BUILD_TASK_STORE" \
        KDA_EXPECT_MMAD_ENGINES="$BUILD_MMAD_ENGINES" \
        KDA_EXPECT_VECTOR_MASK="$BUILD_VECTOR_MASK" \
        KDA_EXPECT_DB_REDUCE="$BUILD_DB_REDUCE" \
        KDA_EXPECT_STAGE_A="$BUILD_STAGE_A" \
        KDA_EXPECT_CUBE_MODE="$BUILD_CUBE_MODE" \
        KDA_EXPECT_STAGE_IO="$BUILD_STAGE_IO" \
        KDA_EXPECT_AIC_DIAGNOSTIC="$BUILD_AIC_DIAGNOSTIC" \
        PYTHONPATH="$RUNTIME_PYTHONPATH" \
        python3 -m pytest --collect-only -q -p no:cacheprovider "$test_file" \
        2>&1 | tee "$RUN_ROOT/logs/kda_collect.log"
    grep -Eq '(37 (tests|items) collected|collected 37 items)' \
        "$RUN_ROOT/logs/kda_collect.log"

    # Fail fast on the first real grouped launch before spending up to twenty
    # minutes in the directed suite. This is the smallest device proof for the
    # shared persistent MMAD event domain and disjoint C workspace tail.
    local preflight_rc
    set +e
    timeout --kill-after=15s 120s env \
        ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" TEST_DEVICE_ID=0 \
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
        KDA_EXPECT_PAIR_GATES="$BUILD_PAIR_GATES" \
        KDA_EXPECT_SHARED_SETUP="$BUILD_SHARED_SETUP" \
        KDA_EXPECT_STAGE_EPILOGUE="$BUILD_STAGE_EPILOGUE" \
        KDA_EXPECT_PAIR_SCRATCH="$BUILD_PAIR_SCRATCH" \
        KDA_EXPECT_TAIL_BLOCKS="$BUILD_TAIL_BLOCKS" \
        KDA_EXPECT_TASK_STORE="$BUILD_TASK_STORE" \
        KDA_EXPECT_MMAD_ENGINES="$BUILD_MMAD_ENGINES" \
        KDA_EXPECT_VECTOR_MASK="$BUILD_VECTOR_MASK" \
        KDA_EXPECT_DB_REDUCE="$BUILD_DB_REDUCE" \
        KDA_EXPECT_STAGE_A="$BUILD_STAGE_A" \
        KDA_EXPECT_CUBE_MODE="$BUILD_CUBE_MODE" \
        KDA_EXPECT_STAGE_IO="$BUILD_STAGE_IO" \
        KDA_EXPECT_AIC_DIAGNOSTIC="$BUILD_AIC_DIAGNOSTIC" \
        PYTHONPATH="$RUNTIME_PYTHONPATH" \
        python3 -m pytest -q -vv -p no:cacheprovider \
        "${test_file}::test_chunk_kda_bwd_intra_safe_gate_grouped_fastpath_bf16" \
        -s 2>&1 | tee \
        "$RUN_ROOT/logs/kda_grouped_preflight_card${PHYSICAL_DEVICE}.log"
    preflight_rc=${PIPESTATUS[0]}
    set -e
    if (( preflight_rc != 0 )); then
        if (( preflight_rc == 124 || preflight_rc == 137 )); then
            echo "[FAIL] grouped BF16 preflight timed out after 120 seconds " \
                 "(exit=$preflight_rc)" >&2
        else
            echo "[FAIL] grouped BF16 preflight failed (exit=$preflight_rc)" >&2
        fi
        return "$preflight_rc"
    fi
    echo "[PASS] grouped BF16 preflight completed"

    timeout --kill-after=30s 1200s env \
        ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" TEST_DEVICE_ID=0 \
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
        KDA_EXPECT_PAIR_GATES="$BUILD_PAIR_GATES" \
        KDA_EXPECT_SHARED_SETUP="$BUILD_SHARED_SETUP" \
        KDA_EXPECT_STAGE_EPILOGUE="$BUILD_STAGE_EPILOGUE" \
        KDA_EXPECT_PAIR_SCRATCH="$BUILD_PAIR_SCRATCH" \
        KDA_EXPECT_TAIL_BLOCKS="$BUILD_TAIL_BLOCKS" \
        KDA_EXPECT_TASK_STORE="$BUILD_TASK_STORE" \
        KDA_EXPECT_MMAD_ENGINES="$BUILD_MMAD_ENGINES" \
        KDA_EXPECT_VECTOR_MASK="$BUILD_VECTOR_MASK" \
        KDA_EXPECT_DB_REDUCE="$BUILD_DB_REDUCE" \
        KDA_EXPECT_STAGE_A="$BUILD_STAGE_A" \
        KDA_EXPECT_CUBE_MODE="$BUILD_CUBE_MODE" \
        KDA_EXPECT_STAGE_IO="$BUILD_STAGE_IO" \
        KDA_EXPECT_AIC_DIAGNOSTIC="$BUILD_AIC_DIAGNOSTIC" \
        PYTHONPATH="$RUNTIME_PYTHONPATH" \
        python3 -m pytest -q -vv -p no:cacheprovider "$test_file" \
        -k 'grouped_fastpath or grouped_dispatch_source_contract or unsafe_target_shape or reference_right_diag_ftz_guard or reference_left_diag_overflow_guard or reference_off_right_cross_block_ftz_guard or reference_grouped_cancellation_guard' \
        -s 2>&1 | tee "$RUN_ROOT/logs/kda_grouped_directed_card${PHYSICAL_DEVICE}.log"

    timeout --kill-after=30s 1800s env \
        ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" TEST_DEVICE_ID=0 \
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
        KDA_EXPECT_PAIR_GATES="$BUILD_PAIR_GATES" \
        KDA_EXPECT_SHARED_SETUP="$BUILD_SHARED_SETUP" \
        KDA_EXPECT_STAGE_EPILOGUE="$BUILD_STAGE_EPILOGUE" \
        KDA_EXPECT_PAIR_SCRATCH="$BUILD_PAIR_SCRATCH" \
        KDA_EXPECT_TAIL_BLOCKS="$BUILD_TAIL_BLOCKS" \
        KDA_EXPECT_TASK_STORE="$BUILD_TASK_STORE" \
        KDA_EXPECT_MMAD_ENGINES="$BUILD_MMAD_ENGINES" \
        KDA_EXPECT_VECTOR_MASK="$BUILD_VECTOR_MASK" \
        KDA_EXPECT_DB_REDUCE="$BUILD_DB_REDUCE" \
        KDA_EXPECT_STAGE_A="$BUILD_STAGE_A" \
        KDA_EXPECT_CUBE_MODE="$BUILD_CUBE_MODE" \
        KDA_EXPECT_STAGE_IO="$BUILD_STAGE_IO" \
        KDA_EXPECT_AIC_DIAGNOSTIC="$BUILD_AIC_DIAGNOSTIC" \
        PYTHONPATH="$RUNTIME_PYTHONPATH" \
        python3 -m pytest -q -vv -p no:cacheprovider "$test_file" -s \
        --junitxml="$RUN_ROOT/logs/kda_full37.xml" \
        2>&1 | tee "$RUN_ROOT/logs/kda_full37_card${PHYSICAL_DEVICE}.log"

    FULL37_XML="$RUN_ROOT/logs/kda_full37.xml" python3 - <<'PY'
import os
import xml.etree.ElementTree as ET

root = ET.parse(os.environ["FULL37_XML"]).getroot()
cases = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "testcase"]
totals = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
for case in cases:
    child_tags = [child.tag.rsplit("}", 1)[-1] for child in case]
    totals["failures"] += child_tags.count("failure")
    totals["errors"] += child_tags.count("error")
    totals["skipped"] += child_tags.count("skipped")
print("full37 JUnit:", totals)
expected = {"tests": 37, "failures": 0, "errors": 0, "skipped": 0}
if totals != expected:
    raise SystemExit(f"full regression is not an exact 37/37 pass: {totals}")
PY

    cat >"$RUN_ROOT/full37.pass" <<EOF
commit=$COMMIT
wheel_sha256=$WHEEL_SHA256
wheel_kda_digest=$WHEEL_KDA_DIGEST
physical_device=$PHYSICAL_DEVICE
cann_env=$BUILD_CANN_ENV
cann_env_sha256=$BUILD_CANN_ENV_SHA256
ascend_home=$BUILD_ASCEND_HOME
cann_version_digest=$BUILD_CANN_VERSION_DIGEST
soc=$BUILD_SOC
python=$BUILD_PYTHON
python_version=$BUILD_PYTHON_VERSION
torch_version=$BUILD_TORCH_VERSION
torch_npu_version=$BUILD_TORCH_NPU_VERSION
validation_manifest_sha256=$VALIDATION_MANIFEST_SHA256
catlass_commit=$BUILD_CATLASS_COMMIT
catlass_tree=$BUILD_CATLASS_TREE
runner_sha256=$BUILD_RUNNER_SHA256
variant=$BUILD_VARIANT_ID
pair_gates=$BUILD_PAIR_GATES
shared_setup=$BUILD_SHARED_SETUP
stage_epilogue=$BUILD_STAGE_EPILOGUE
pair_scratch=$BUILD_PAIR_SCRATCH
tail_blocks=$BUILD_TAIL_BLOCKS
task_store=$BUILD_TASK_STORE
mmad_engines=$BUILD_MMAD_ENGINES
vector_mask=$BUILD_VECTOR_MASK
db_reduce=$BUILD_DB_REDUCE
stage_a=$BUILD_STAGE_A
cube_mode=$BUILD_CUBE_MODE
stage_io=$BUILD_STAGE_IO
aic_diagnostic=$BUILD_AIC_DIAGNOSTIC
tests=37
failures=0
errors=0
skipped=0
EOF
    sha256sum "$WHEEL" >"$RUN_ROOT/full37_wheel.sha256"
    echo "[PASS] directed grouped cases and full 37-item regression"
}

profile_metric() {
    local metric=$1 output=$2 warmup=$3 repeat=$4 perf_script=$5
    shift 5
    local -a perf_args=("$@")
    mkdir -p "$output"
    timeout --kill-after=30s 1800s env \
        ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" \
        KDA_BUILD_VARIANT="$BUILD_VARIANT_ID" \
        KDA_BUILD_PAIR_GATES="$BUILD_PAIR_GATES" \
        KDA_BUILD_SHARED_SETUP="$BUILD_SHARED_SETUP" \
        KDA_BUILD_STAGE_EPILOGUE="$BUILD_STAGE_EPILOGUE" \
        KDA_BUILD_PAIR_SCRATCH="$BUILD_PAIR_SCRATCH" \
        KDA_BUILD_TAIL_BLOCKS="$BUILD_TAIL_BLOCKS" \
        KDA_BUILD_TASK_STORE="$BUILD_TASK_STORE" \
        KDA_BUILD_MMAD_ENGINES="$BUILD_MMAD_ENGINES" \
        KDA_BUILD_VECTOR_MASK="$BUILD_VECTOR_MASK" \
        KDA_BUILD_DB_REDUCE="$BUILD_DB_REDUCE" \
        KDA_BUILD_STAGE_A="$BUILD_STAGE_A" \
        KDA_BUILD_CUBE_MODE="$BUILD_CUBE_MODE" \
        KDA_BUILD_STAGE_IO="$BUILD_STAGE_IO" \
        PYTHONPATH="$RUNTIME_PYTHONPATH" \
        msprof --output="$output" --ai-core=on --aic-metrics="$metric" \
        --task-time=on --ascendcl=on \
        python3 "$perf_script" "${perf_args[@]}" --warmup "$warmup" --repeat "$repeat" \
        --manifest-out "$output/launch_manifest.json"
}

verify_variant_manifest() {
    local manifest=$1
    BUILD_VARIANT_ID="$BUILD_VARIANT_ID" BUILD_PAIR_GATES="$BUILD_PAIR_GATES" \
    BUILD_SHARED_SETUP="$BUILD_SHARED_SETUP" \
    BUILD_STAGE_EPILOGUE="$BUILD_STAGE_EPILOGUE" \
    BUILD_PAIR_SCRATCH="$BUILD_PAIR_SCRATCH" \
    BUILD_TAIL_BLOCKS="$BUILD_TAIL_BLOCKS" \
    BUILD_TASK_STORE="$BUILD_TASK_STORE" \
    BUILD_MMAD_ENGINES="$BUILD_MMAD_ENGINES" \
    BUILD_VECTOR_MASK="$BUILD_VECTOR_MASK" \
    BUILD_DB_REDUCE="$BUILD_DB_REDUCE" \
    BUILD_STAGE_A="$BUILD_STAGE_A" \
    BUILD_CUBE_MODE="$BUILD_CUBE_MODE" \
    BUILD_STAGE_IO="$BUILD_STAGE_IO" python3 - "$manifest" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
expected = {
    "build_variant": os.environ["BUILD_VARIANT_ID"],
    "pair_gates": os.environ["BUILD_PAIR_GATES"],
    "shared_setup": os.environ["BUILD_SHARED_SETUP"],
    "stage_epilogue": os.environ["BUILD_STAGE_EPILOGUE"],
    "pair_scratch": os.environ["BUILD_PAIR_SCRATCH"],
    "tail_blocks": os.environ["BUILD_TAIL_BLOCKS"],
    "task_store": os.environ["BUILD_TASK_STORE"],
    "mmad_engines": os.environ["BUILD_MMAD_ENGINES"],
    "vector_mask": os.environ["BUILD_VECTOR_MASK"],
    "db_reduce": os.environ["BUILD_DB_REDUCE"],
    "stage_a": os.environ["BUILD_STAGE_A"],
    "cube_mode": os.environ["BUILD_CUBE_MODE"],
    "stage_io": os.environ["BUILD_STAGE_IO"],
}
actual = {name: manifest.get(name) for name in expected}
if actual != expected:
    raise SystemExit(f"launch variant mismatch: expected={expected}, actual={actual}")
print(f"launch variant manifest: {actual}")
PY
}

run_profile() {
    [[ -n "${RUN_ROOT:-}" ]] || load_state
    [[ -f "$RUN_ROOT/full37.pass" ]] || {
        echo "[FAIL] run --mode test successfully before profiling this wheel" >&2
        exit 1
    }
    grep -Fxq "commit=$COMMIT" "$RUN_ROOT/full37.pass"
    grep -Fxq "wheel_sha256=$WHEEL_SHA256" "$RUN_ROOT/full37.pass"
    grep -Fxq "wheel_kda_digest=$WHEEL_KDA_DIGEST" "$RUN_ROOT/full37.pass"
    grep -Fxq "physical_device=$PHYSICAL_DEVICE" "$RUN_ROOT/full37.pass" || {
        echo "[FAIL] profiling device differs from the full37 device" >&2
        exit 1
    }
    grep -Fxq 'tests=37' "$RUN_ROOT/full37.pass"
    grep -Fxq 'failures=0' "$RUN_ROOT/full37.pass"
    grep -Fxq 'errors=0' "$RUN_ROOT/full37.pass"
    grep -Fxq 'skipped=0' "$RUN_ROOT/full37.pass"
    grep -Fxq "cann_env=$BUILD_CANN_ENV" "$RUN_ROOT/full37.pass"
    grep -Fxq "cann_env_sha256=$BUILD_CANN_ENV_SHA256" "$RUN_ROOT/full37.pass"
    grep -Fxq "ascend_home=$BUILD_ASCEND_HOME" "$RUN_ROOT/full37.pass"
    grep -Fxq "cann_version_digest=$BUILD_CANN_VERSION_DIGEST" "$RUN_ROOT/full37.pass"
    grep -Fxq "soc=$BUILD_SOC" "$RUN_ROOT/full37.pass"
    grep -Fxq "python=$BUILD_PYTHON" "$RUN_ROOT/full37.pass"
    grep -Fxq "python_version=$BUILD_PYTHON_VERSION" "$RUN_ROOT/full37.pass"
    grep -Fxq "torch_version=$BUILD_TORCH_VERSION" "$RUN_ROOT/full37.pass"
    grep -Fxq "torch_npu_version=$BUILD_TORCH_NPU_VERSION" "$RUN_ROOT/full37.pass"
    grep -Fxq "validation_manifest_sha256=$VALIDATION_MANIFEST_SHA256" \
        "$RUN_ROOT/full37.pass"
    grep -Fxq "catlass_commit=$BUILD_CATLASS_COMMIT" "$RUN_ROOT/full37.pass"
    grep -Fxq "catlass_tree=$BUILD_CATLASS_TREE" "$RUN_ROOT/full37.pass"
    grep -Fxq "runner_sha256=$BUILD_RUNNER_SHA256" "$RUN_ROOT/full37.pass"
    grep -Fxq "variant=$BUILD_VARIANT_ID" "$RUN_ROOT/full37.pass"
    grep -Fxq "pair_gates=$BUILD_PAIR_GATES" "$RUN_ROOT/full37.pass"
    grep -Fxq "shared_setup=$BUILD_SHARED_SETUP" "$RUN_ROOT/full37.pass"
    grep -Fxq "stage_epilogue=$BUILD_STAGE_EPILOGUE" "$RUN_ROOT/full37.pass"
    grep -Fxq "pair_scratch=$BUILD_PAIR_SCRATCH" "$RUN_ROOT/full37.pass"
    grep -Fxq "tail_blocks=$BUILD_TAIL_BLOCKS" "$RUN_ROOT/full37.pass"
    grep -Fxq "task_store=$BUILD_TASK_STORE" "$RUN_ROOT/full37.pass"
    grep -Fxq "mmad_engines=$BUILD_MMAD_ENGINES" "$RUN_ROOT/full37.pass"
    grep -Fxq "vector_mask=$BUILD_VECTOR_MASK" "$RUN_ROOT/full37.pass"
    grep -Fxq "db_reduce=$BUILD_DB_REDUCE" "$RUN_ROOT/full37.pass"
    grep -Fxq "stage_a=$BUILD_STAGE_A" "$RUN_ROOT/full37.pass"
    grep -Fxq "cube_mode=$BUILD_CUBE_MODE" "$RUN_ROOT/full37.pass"
    grep -Fxq "stage_io=$BUILD_STAGE_IO" "$RUN_ROOT/full37.pass"
    grep -Fxq "aic_diagnostic=$BUILD_AIC_DIAGNOSTIC" "$RUN_ROOT/full37.pass"
    runtime_env
    sha256sum -c "$RUN_ROOT/full37_wheel.sha256"
    create_runtime_install profile
    command -v msprof >/dev/null || { echo "[FAIL] msprof not found" >&2; exit 1; }
    local msprof_real cann_bundle_root
    msprof_real="$(readlink -f "$(command -v msprof)")"
    cann_bundle_root="$(cd "$(dirname "$CANN_ENV")/.." && pwd -P)"
    case "$msprof_real/" in
        "$cann_bundle_root/"*) ;;
        *) echo "[FAIL] msprof is outside the selected CANN bundle: $msprof_real" >&2; exit 1 ;;
    esac

    local perf_script stamp profile_group pipe_dir op_csv
    local e2e_manifest e2e_log pipe_log profile_summary profile_json
    local -a perf_args=(--device 0 --batch 1 --seqlen 8192 --heads 32 \
        --head-dim 128 --chunk-size 64 --safe-gate)
    perf_script="$WHEEL_SRC/torch_custom/fla_npu/test/benchmark_npu_chunk_kda_bwd_intra.py"
    stamp="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_ROOT/msprof"
    profile_group="$(mktemp -d \
        "$RUN_ROOT/msprof/PROF_GROUP_${stamp}_${BUILD_VARIANT_ID}_XXXXXX")"
    pipe_dir="$profile_group/PROF_PipeUtilization"
    e2e_manifest="$profile_group/e2e_launch_manifest.json"
    e2e_log="$profile_group/kda_e2e_8192_card${PHYSICAL_DEVICE}.log"
    pipe_log="$profile_group/kda_msprof_pipe_card${PHYSICAL_DEVICE}.log"
    profile_summary="$profile_group/kda_profile_summary.txt"
    profile_json="$profile_group/kda_profile_summary.json"

    timeout --kill-after=30s 900s env \
        ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" \
        KDA_BUILD_VARIANT="$BUILD_VARIANT_ID" \
        KDA_BUILD_PAIR_GATES="$BUILD_PAIR_GATES" \
        KDA_BUILD_SHARED_SETUP="$BUILD_SHARED_SETUP" \
        KDA_BUILD_STAGE_EPILOGUE="$BUILD_STAGE_EPILOGUE" \
        KDA_BUILD_PAIR_SCRATCH="$BUILD_PAIR_SCRATCH" \
        KDA_BUILD_TAIL_BLOCKS="$BUILD_TAIL_BLOCKS" \
        KDA_BUILD_TASK_STORE="$BUILD_TASK_STORE" \
        KDA_BUILD_MMAD_ENGINES="$BUILD_MMAD_ENGINES" \
        KDA_BUILD_VECTOR_MASK="$BUILD_VECTOR_MASK" \
        KDA_BUILD_DB_REDUCE="$BUILD_DB_REDUCE" \
        KDA_BUILD_STAGE_A="$BUILD_STAGE_A" \
        KDA_BUILD_CUBE_MODE="$BUILD_CUBE_MODE" \
        KDA_BUILD_STAGE_IO="$BUILD_STAGE_IO" \
        PYTHONPATH="$RUNTIME_PYTHONPATH" \
        python3 "$perf_script" "${perf_args[@]}" --warmup 3 --repeat 10 \
        --manifest-out "$e2e_manifest" \
        2>&1 | tee "$e2e_log"
    verify_variant_manifest "$e2e_manifest"
    grep -Fxq 'shape=B1_T8192_H32_K128_BT64_BF16_safe' \
        "$e2e_log"
    grep -Fxq 'warmup=3, repeat=10' "$e2e_log"
    grep -Fxq "cube_mode=$BUILD_CUBE_MODE" "$e2e_log"
    grep -Fxq "variant=$BUILD_VARIANT_ID, pair_gates=$BUILD_PAIR_GATES, shared_setup=$BUILD_SHARED_SETUP, stage_epilogue=$BUILD_STAGE_EPILOGUE, pair_scratch=$BUILD_PAIR_SCRATCH, tail_blocks=$BUILD_TAIL_BLOCKS, task_store=$BUILD_TASK_STORE, mmad_engines=$BUILD_MMAD_ENGINES, vector_mask=$BUILD_VECTOR_MASK, db_reduce=$BUILD_DB_REDUCE, stage_a=$BUILD_STAGE_A, stage_io=$BUILD_STAGE_IO" \
        "$e2e_log"

    # Warmup and ten measured launches stay in the same profiler process.
    profile_metric PipeUtilization "$pipe_dir" 3 10 "$perf_script" \
        "${perf_args[@]}" 2>&1 | \
        tee "$pipe_log"
    op_csv="$(find "$pipe_dir" -type f -name 'op_summary_*.csv' -print | sort | tail -1)"
    [[ -f "$op_csv" ]] || { echo "[FAIL] op_summary not found" >&2; exit 1; }
    local launch_manifest="$pipe_dir/launch_manifest.json"
    [[ -f "$launch_manifest" ]] || { echo "[FAIL] profile launch manifest not found" >&2; exit 1; }

    local analyzer=(python3 "$WHEEL_SRC/scripts/analyze_chunk_kda_bwd_intra_profile.py"
        "$op_csv" --target-ms 4.0 --baseline-ms 48.660472 \
        --discard-first 3 --min-rows 10 --expected-rows 10 \
        --expected-device-id "$PHYSICAL_DEVICE" --launch-manifest "$launch_manifest" \
        --expected-build-variant "$BUILD_VARIANT_ID" \
        --expected-pair-gates "$BUILD_PAIR_GATES" \
        --expected-shared-setup "$BUILD_SHARED_SETUP" \
        --expected-stage-epilogue "$BUILD_STAGE_EPILOGUE" \
        --expected-pair-scratch "$BUILD_PAIR_SCRATCH" \
        --expected-tail-blocks "$BUILD_TAIL_BLOCKS" \
        --expected-task-store "$BUILD_TASK_STORE" \
        --expected-mmad-engines "$BUILD_MMAD_ENGINES" \
        --expected-vector-mask "$BUILD_VECTOR_MASK" \
        --expected-db-reduce "$BUILD_DB_REDUCE" \
        --expected-stage-a "$BUILD_STAGE_A" \
        --expected-cube-mode "$BUILD_CUBE_MODE" \
        --expected-stage-io "$BUILD_STAGE_IO" \
        --require-target-shape --require-mixed)
    if $FINAL_GATE; then
        analyzer+=(--require-under-target)
    fi
    "${analyzer[@]}" | tee "$profile_summary"
    "${analyzer[@]}" --json >"$profile_json"

    local profile_complete=true
    if $FULL_METRICS; then
        local metric metric_dir metric_csv
        for metric in ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache \
            ResourceConflictRatio; do
            metric_dir="$profile_group/PROF_${metric}"
            profile_metric "$metric" "$metric_dir" 3 1 "$perf_script" \
                "${perf_args[@]}" >"$profile_group/PROF_${metric}.log" 2>&1
            metric_csv="$(find "$metric_dir" -type f -name 'op_summary_*.csv' \
                -print | sort | tail -1)"
            [[ -f "$metric_csv" ]] || {
                echo "[FAIL] $metric op_summary not found" >&2
                exit 1
            }
            python3 "$WHEEL_SRC/scripts/analyze_chunk_kda_bwd_intra_profile.py" \
                "$metric_csv" --discard-first 3 --expected-rows 1 \
                --expected-device-id "$PHYSICAL_DEVICE" \
                --launch-manifest "$metric_dir/launch_manifest.json" --require-target-shape \
                --expected-build-variant "$BUILD_VARIANT_ID" \
                --expected-pair-gates "$BUILD_PAIR_GATES" \
                --expected-shared-setup "$BUILD_SHARED_SETUP" \
                --expected-stage-epilogue "$BUILD_STAGE_EPILOGUE" \
                --expected-pair-scratch "$BUILD_PAIR_SCRATCH" \
                --expected-tail-blocks "$BUILD_TAIL_BLOCKS" \
                --expected-task-store "$BUILD_TASK_STORE" \
                --expected-mmad-engines "$BUILD_MMAD_ENGINES" \
                --expected-vector-mask "$BUILD_VECTOR_MASK" \
                --expected-db-reduce "$BUILD_DB_REDUCE" \
                --expected-stage-a "$BUILD_STAGE_A" \
                --expected-cube-mode "$BUILD_CUBE_MODE" \
                --expected-stage-io "$BUILD_STAGE_IO" \
                --require-mixed >"$profile_group/kda_profile_${metric}_summary.txt"
        done
        local sample_dir="$profile_group/PROF_Sample"
        mkdir -p "$sample_dir"
        set +e
        timeout --kill-after=30s 1800s env \
            ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" \
            KDA_BUILD_VARIANT="$BUILD_VARIANT_ID" \
            KDA_BUILD_PAIR_GATES="$BUILD_PAIR_GATES" \
            KDA_BUILD_SHARED_SETUP="$BUILD_SHARED_SETUP" \
            KDA_BUILD_STAGE_EPILOGUE="$BUILD_STAGE_EPILOGUE" \
            KDA_BUILD_PAIR_SCRATCH="$BUILD_PAIR_SCRATCH" \
            KDA_BUILD_TAIL_BLOCKS="$BUILD_TAIL_BLOCKS" \
            KDA_BUILD_TASK_STORE="$BUILD_TASK_STORE" \
            KDA_BUILD_MMAD_ENGINES="$BUILD_MMAD_ENGINES" \
            KDA_BUILD_VECTOR_MASK="$BUILD_VECTOR_MASK" \
            KDA_BUILD_DB_REDUCE="$BUILD_DB_REDUCE" \
            KDA_BUILD_STAGE_A="$BUILD_STAGE_A" \
            KDA_BUILD_CUBE_MODE="$BUILD_CUBE_MODE" \
            KDA_BUILD_STAGE_IO="$BUILD_STAGE_IO" \
            PYTHONPATH="$RUNTIME_PYTHONPATH" \
            msprof --output="$sample_dir" --ai-core=on \
            --aic-metrics=PipeUtilization --aic-mode=sample-based --aic-freq=100 \
            --task-time=on --ascendcl=on \
            python3 "$perf_script" "${perf_args[@]}" --warmup 3 --repeat 1 \
            --manifest-out "$sample_dir/launch_manifest.json" \
            >"$sample_dir/msprof.log" 2>&1
        local sample_rc=$?
        set -e
        local sample_manifest="$sample_dir/launch_manifest.json"
        local sample_manifest_ok=true
        if [[ -f "$sample_manifest" ]]; then
            verify_variant_manifest "$sample_manifest"
        else
            sample_manifest_ok=false
        fi
        local -a sample_dbs=()
        mapfile -t sample_dbs < <(find "$sample_dir" -type f -name 'aicore.db' \
            -size +0c -print | sort)
        if [[ ${#sample_dbs[@]} -gt 0 ]]; then
            sha256sum "${sample_dbs[@]}" | \
                tee "$profile_group/kda_profile_sample_aicore_db.sha256"
        fi
        if [[ $sample_rc -ne 0 || ${#sample_dbs[@]} -eq 0 || \
              "$sample_manifest_ok" != true ]]; then
            if $FINAL_GATE; then
                echo "[FAIL] sample-based profile is incomplete: rc=$sample_rc, db_count=${#sample_dbs[@]}, manifest=$sample_manifest_ok" >&2
                exit 1
            fi
            profile_complete=false
            echo "[WARN] sample-based profile is incomplete: rc=$sample_rc, db_count=${#sample_dbs[@]}, manifest=$sample_manifest_ok"
        fi
    fi

    local performance_status artifact_listing evidence_file evidence_digest identity_file
    local full37_digest pipe_csv_digest pipe_manifest_digest e2e_manifest_digest digest
    local -a discovered_artifacts=() profile_artifacts=()
    performance_status="$(awk -F= '$1 == "performance_target" {print $2}' \
        "$profile_summary" | tail -1)"
    [[ "$performance_status" == PASS || "$performance_status" == NOT_MET ]] || {
        echo "[FAIL] performance status missing from analyzer output" >&2
        exit 1
    }
    artifact_listing="$(find "$profile_group" -type f \
        \( -name 'op_summary_*.csv' -o -name '*launch_manifest.json' \
           -o -name 'aicore.db' -o -name 'kda_profile_*_summary.txt' \
        \) -print | sort)"
    [[ -n "$artifact_listing" ]] || {
        echo "[FAIL] no profile artifacts available for identity record" >&2
        exit 1
    }
    mapfile -t discovered_artifacts <<<"$artifact_listing"
    profile_artifacts=("$e2e_log" "$pipe_log" "$profile_summary" "$profile_json" \
        "${discovered_artifacts[@]}")
    evidence_file="$profile_group/profile_evidence.sha256"
    sha256sum "$WHEEL" "$RUN_ROOT/full37.pass" "$VALIDATION_MANIFEST" \
        "${profile_artifacts[@]}" >"$evidence_file"
    sha256sum --check --quiet --strict "$evidence_file"
    evidence_digest="$(sha256sum "$evidence_file" | awk '{print $1}')"
    full37_digest="$(sha256sum "$RUN_ROOT/full37.pass" | awk '{print $1}')"
    pipe_csv_digest="$(sha256sum "$op_csv" | awk '{print $1}')"
    pipe_manifest_digest="$(sha256sum "$launch_manifest" | awk '{print $1}')"
    e2e_manifest_digest="$(sha256sum "$e2e_manifest" | awk '{print $1}')"
    for digest in "$evidence_digest" "$full37_digest" "$pipe_csv_digest" \
        "$pipe_manifest_digest" "$e2e_manifest_digest"; do
        [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
            echo "[FAIL] invalid SHA256 while writing profile identity: $digest" >&2
            exit 1
        }
    done
    identity_file="$profile_group/profile_identity.txt"
    cat >"$identity_file" <<EOF
commit=$COMMIT
wheel_sha256=$WHEEL_SHA256
wheel_kda_digest=$WHEEL_KDA_DIGEST
physical_device=$PHYSICAL_DEVICE
cann_env=$BUILD_CANN_ENV
cann_env_sha256=$BUILD_CANN_ENV_SHA256
cann_version_digest=$BUILD_CANN_VERSION_DIGEST
validation_manifest_sha256=$VALIDATION_MANIFEST_SHA256
catlass_commit=$BUILD_CATLASS_COMMIT
catlass_tree=$BUILD_CATLASS_TREE
runner_sha256=$BUILD_RUNNER_SHA256
variant=$BUILD_VARIANT_ID
pair_gates=$BUILD_PAIR_GATES
shared_setup=$BUILD_SHARED_SETUP
stage_epilogue=$BUILD_STAGE_EPILOGUE
pair_scratch=$BUILD_PAIR_SCRATCH
tail_blocks=$BUILD_TAIL_BLOCKS
task_store=$BUILD_TASK_STORE
mmad_engines=$BUILD_MMAD_ENGINES
vector_mask=$BUILD_VECTOR_MASK
db_reduce=$BUILD_DB_REDUCE
stage_a=$BUILD_STAGE_A
cube_mode=$BUILD_CUBE_MODE
stage_io=$BUILD_STAGE_IO
aic_diagnostic=$BUILD_AIC_DIAGNOSTIC
full37_sha256=$full37_digest
pipe_csv=$op_csv
pipe_csv_sha256=$pipe_csv_digest
pipe_manifest_sha256=$pipe_manifest_digest
e2e_manifest_sha256=$e2e_manifest_digest
profile_evidence_sha256=$evidence_digest
profile_complete=$profile_complete
performance_target=$performance_status
final_gate=$FINAL_GATE
EOF
    if $profile_complete; then
        local pass_tmp="$profile_group/.profile_evidence.pass.tmp.$$"
        cp "$identity_file" "$pass_tmp"
        mv -f "$pass_tmp" "$profile_group/profile_evidence.pass"
        echo "[PASS] requested profile evidence is complete: $profile_group"
        echo "[PASS] profile identity: $profile_group/profile_evidence.pass"
    else
        echo "[WARN] ten-sample PipeUtilization passed, but requested full metrics are incomplete: $profile_group"
        echo "[WARN] incomplete profile identity: $identity_file"
    fi
}

case "$MODE" in
    build)
        run_build
        ;;
    test)
        load_state
        run_test
        ;;
    profile)
        load_state
        run_profile
        ;;
    all)
        run_build
        run_test
        run_profile
        ;;
esac
