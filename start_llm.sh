log_dir="./results/bandit"

time_period=100000
n_features=512
n_arms=1
n_expe=1
NpS=8
noise_dim=4
action_noise=sp
update_noise=pm
buffer_noise=sp
batch_size=32
update_start=20
update_num=1
update_freq=10
lr=1e-4
weight_decay=0.1
prior_scale=0.2
posterior_scale=0.1
action_num=2
fine_tune=1

llm_name=gpt2
# llm_name=pythia-14m

model_type=ensemble++
# model_type=ensemble+
# model_type=linear

cuda_id=$2

seed=2020
for i in $(seq 4)
do
    export CUDA_VISIBLE_DEVICES=${cuda_id}
    tag=$(date "+%Y%m%d%H%M%S")
    python3 -m scripts.run_llm --game='hatespeech' --seed=${seed} --n-features=${n_features} --n-arms=${n_arms} \
        --noise-dim=${noise_dim} --NpS=${NpS} --model-type=${model_type} --llm-name=${llm_name} \
        --weight-decay=${weight_decay} --batch-size=${batch_size} --lr=${lr} \
        --update-start=${update_start} --update-num=${update_num} --update-freq=${update_freq} \
        --prior-scale=${prior_scale} --posterior-scale=${posterior_scale} \
        --action-num=${action_num} --fine-tune=${fine_tune} \
        --action-noise=${action_noise} --update-noise=${update_noise} --buffer-noise=${buffer_noise} \
        --time-period=${time_period} --n-expe=${n_expe} --log-dir=${log_dir} \
        > ~/logs/${game}_${tag}.out 2> ~/logs/${game}_${tag}.err &
    echo "run $model_type hatespeech $cuda_id $seed $tag"
    let seed=$seed+1
    let cuda_id=$cuda_id+1
    sleep 1.0
done
let cuda_id=$cuda_id+1

# python taiji/run_gpu.py
# ps -ef | grep llm | awk '{print $2}'| xargs kill -9
