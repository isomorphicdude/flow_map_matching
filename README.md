# flow_map_matching

Minimal implementation of the flow map matching method (https://arxiv.org/abs/2406.07507) in `jax`.

Note: this is an unofficial implementation and does not recreate the exact experiments of the paper, but does contain implementations of the associated loss functions and basic training loops.

Also contains a modified version of the HuggingFace Diffusers UNet implementation to allow for two times as needed by the flow map formalism, along with a minimal stochastic interpolant implementation.

## Running

The main driver script should be ran from `py/launchers`. For example, a reasonable configuration to train on the MNIST dataset looks something like the following.

```bash
cd py/launchers
python learn.py \
    --n_epochs 10000 \
    --bs 256 \
    --plot_bs 5 \
    --visual_freq 1000 \
    --save_freq 10000 \
    --shuffle_dataset 1 \
    --overfit 0 \
    --conditional 1 \
    --class_dropout 0.0 \
    --box_anneal 0 \
    --diagonal_anneal 1 \
    --anneal_steps 7500 \
    --distill_steps 0 \
    --distill_delta 0 \
    --n 100000 \
    --d 784 \
    --tmin 0 \
    --tmax 1.0 \
    --n_neurons 0 \
    --n_hidden 0 \
    --act 'swish' \
    --learning_rate 0.001 \
    --decay_steps 1500000 \
    --warmup_steps 0 \
    --loss_type 'lagrangian' \
    --base 'gaussian' \
    --gaussian_scale 'adaptive' \
    --target 'mnist' \
    --device_type 'gpu' \
    --wandb_name '<name>' \
    --wandb_entity '<entity>' \
    --wandb_project '<project>' \
    --output_name '<name>' \
    --output_folder 'results' \
    --dataset_folder 'datasets' \
```
