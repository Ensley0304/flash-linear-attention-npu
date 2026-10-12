set -e
cd /data/wys/kda-finalize-a2-stage9-20261012
tag=$1
for i in $(seq 1 180); do
    test -f bound-opt/r2-model.status && break
    sleep 5
done
test -f bound-opt/r2-model.status
if ! test -f bound-opt/$tag-check.status; then
    bash bound-opt/run.sh "$tag" check
fi
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
python -u bound-opt/model.py --library bound-opt/$tag-capture.so --tag $tag-model-capture --gold r2-model.json --device 3 > bound-opt/$tag-model-capture.log 2>&1
python -u bound-opt/model.py --library bound-opt/$tag-perf.so --tag $tag-model-perf --gold r2-model.json --device 3 --perf > bound-opt/$tag-model-perf.log 2>&1
bash bound-opt/run.sh r2-final profile --library-tag r2-rebuilt
bash bound-opt/run.sh $tag-final profile --library-tag "$tag"
echo PASS > bound-opt/$tag-final.status
