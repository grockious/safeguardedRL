#!/bin/sh

# args
noise="0"
exploration_loss_coeff_1="0.003"
exploration_loss_coeff_2="0.003"
load_optimizer_1="False"
load_optimizer_2="False"
use_rnn="True"
log_rollout_violations="False"
num_envs_per_worker="16"
wandb_user="0-flails_saddles"


for i in 1 2 3 4 5 6 7 8 9 10
do
    prefix="baseline"
    violation_penalty="0"
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=40000000 --violation_penalty=$violation_penalty --safeguard=-1 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15

    prefix="zero_shot"
    violation_penalty="0"
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=40000000 --violation_penalty=$violation_penalty --safeguard=2 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15

    prefix="safe_no_penalty"
    violation_penalty="0"
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=2500000 --violation_penalty=$violation_penalty --safeguard=0 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=15000000 --violation_penalty=$violation_penalty --exploration_loss_coeff=$exploration_loss_coeff_1 --load_optimizer=$load_optimizer_1 --noise_mult=$noise --safeguard=1 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=40000000 --violation_penalty=$violation_penalty --exploration_loss_coeff=$exploration_loss_coeff_1 --load_optimizer=$load_optimizer_2 --noise_mult=$noise --safeguard=2 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15

    prefix="safe_half_penalty"
    violation_penalty="0.5"
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=2500000 --violation_penalty=$violation_penalty --safeguard=0 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=15000000 --violation_penalty=$violation_penalty --exploration_loss_coeff=$exploration_loss_coeff_1 --load_optimizer=$load_optimizer_1 --noise_mult=$noise --safeguard=1 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=40000000 --violation_penalty=$violation_penalty --exploration_loss_coeff=$exploration_loss_coeff_1 --load_optimizer=$load_optimizer_2 --noise_mult=$noise --safeguard=2 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15

    prefix="safe_full_penalty"
    violation_penalty="1"
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=2500000 --violation_penalty=$violation_penalty --safeguard=0 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=15000000 --violation_penalty=$violation_penalty --exploration_loss_coeff=$exploration_loss_coeff_1 --load_optimizer=$load_optimizer_1 --noise_mult=$noise --safeguard=1 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15
    python3 -m  sf_examples.vizdoom.train_vizdoom --env=safe_rl --experiment=SafeRl$prefix$i --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=$num_envs_per_worker --use_rnn=$use_rnn --train_for_env_steps=40000000 --violation_penalty=$violation_penalty --exploration_loss_coeff=$exploration_loss_coeff_1 --load_optimizer=$load_optimizer_2 --noise_mult=$noise --safeguard=2 --prefix=$prefix --main_level=1 --trial=$i --with_wandb=True --wandb_user=$wandb_user --wandb_tags $prefix
    sleep 15

done
