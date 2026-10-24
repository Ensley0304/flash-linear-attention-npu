#!/usr/bin/env bash
set -euo pipefail
device=${1:-2}
output=${2:?provide an output directory}
plugin=$(cd "$(dirname "$0")/.." && pwd)
: "${KDA_FINALIZE_LIBRARY:?set the freshly built performance library}"
if [[ -e "$output" ]]; then
    echo "Use a fresh output directory: $output" >&2
    exit 1
fi
mkdir -p "$output"
sha256sum "$KDA_FINALIZE_LIBRARY" "$plugin/gen_chunk_kda_bwd_finalize.py" \
    "$plugin/executor_chunk_kda_bwd_finalize.py" "$plugin/scripts/reference.py" \
    "$plugin/atk_chunk_kda_bwd_finalize.json" > "$output/source_sha256.txt"
for round in 0 1 2; do
    export KDA_SEED_ROUND=$round
    python -m atk node --name npu_dut --backend npu --devices "$device" --output_path "$output/round-$round" \
        node --name cpu_ref --backend cpu --output_path "$output/round-$round" \
        task -c "$plugin/atk_chunk_kda_bwd_finalize.json" --task accuracy --bm_device cpu \
        -p "$plugin/executor_chunk_kda_bwd_finalize.py" -cp -sp -to 14400 \
        > "$output/round-$round.log" 2>&1
done
python "$plugin/scripts/check_results.py" "$output"
