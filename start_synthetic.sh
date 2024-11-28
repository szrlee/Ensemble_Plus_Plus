log_dir="./results/bandit"

n_context=1
time_period=100000
n_features=100
n_arms=50
NpS=16
noise_dim=8
action_noise=sp
update_noise=pm
buffer_noise=sp
method=Ensemble++
# method=EpiNet
# method=Ensemble+
# method=NeuralUCB

cuda_id=0
game=Synthetic-v1 # Synthetic-v1 Synthetic-v2 Synthetic-v4 Synthetic-v5 Synthetic-v6 FreqRusso Russo

export CUDA_VISIBLE_DEVICES=${cuda_id}
seed=0
for i in $(seq 5)
do
    tag=$(date "+%Y%m%d%H%M%S")
    python -m scripts.run_nonlinear --game=${game} --method=${method} --seed=${seed} \
        --n-context=${n_context} --time-period=${time_period} \
        --n-features=${n_features} --n-arms=${n_arms} \
        --noise-dim=${noise_dim} --NpS=${NpS} \
        --action-noise=${action_noise} --update-noise=${update_noise} --buffer-noise=${buffer_noise} \
        --log-dir=${log_dir} \
        > ~/logs/${game}_${tag}.out 2> ~/logs/${game}_${tag}.err &
    echo "run $method $game $cuda_id $seed $tag"
    let seed=$seed+1
    sleep 1.0
done
let cuda_id=$cuda_id+1


python taiji/run_gpu.py
# ps -ef | grep hyper | awk '{print $2}'| xargs kill -9
