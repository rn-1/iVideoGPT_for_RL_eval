import torch
import argparse
from ivideogpt.data import *
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms

from train_policy import DinoPolicy

from transformers import AutoModelForCausalLM, AutoConfig

from ivideogpt.vq_model import CompressiveVQModel
from ivideogpt.transformer import HeadModelWithAction

from world_model_gym import WorldModelEnv

from safetensors.torch import load_file

from inference.utils import NPZParser

import numpy as np
import torch
import os
import imageio

mode = "policy"
ctx_len = 2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_world_model(args):
    # we use args directly here for simplicity's sake
    # LOAD TOKENIZER
    tokenizer = CompressiveVQModel.from_pretrained(
        args.tkn,
        subfolder=None,
        low_cpu_mem_usage=False).to(device)
    tokenizer.eval()


    # LOAD TRANSFORMER
    perlude_tokens_num = (256) # TODO where is context lenght and segment length used???
    tokens_per_dyna = 16

    state_dict_path = args.trm
    state_dict = load_file(os.path.join(state_dict_path, 'model.safetensors'))

    config = AutoConfig.from_pretrained(
        "/home/ryannene/iVideoGPT/configs/llama/config.json",
        trust_remote_code=True,
    )
    config.attention_dropout = 0.1
    config.vocab_size = 2 + 8192 + 8192 # vq embeddings, dyna embeddings, and special for action cond

    transformer = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    transformer = HeadModelWithAction(transformer, action_dim=5, prelude_tokens_num=perlude_tokens_num,
        tokens_num_per_dyna=tokens_per_dyna, context=2,
        segment_length=2, model_type='llama',
        reward_prediction=False, action_recon=None
    )# check if this is okay
    transformer.load_state_dict(state_dict, strict=True)
    transformer.eval()
    transformer.to(device)

    return transformer, tokenizer

def load_policy(args):

    model = DinoPolicy()
    model.load_state_dict(torch.load(args.policy)) 
    model.to(device)
    model.eval()
    return model

def predict_step(tokenizer, transformer, frames, seg_length, actions = None):

    pixel_values = input.to(device, non_blocking=True).unsqueeze(0)
    actions = actions.to(device, non_blocking=True) # we will always have this

    tokens, labels = tokenizer.tokenize(pixel_values, ctx_len)
    gen_input = tokens[:, :ctx_len * (16 * 16 + 1)]  # TODO: magic number

    # predict future frames
    max_new_tokens = (1 + 4 * 4) # (segment_length - context_length) + 1
    gen_kwargs = {
        'do_sample': True,
        'temperature': 1.0,
        'top_k': 100,
        'max_new_tokens': max_new_tokens,
    }
    generated_tokens = model.generate(
        gen_input,
        **gen_kwargs,
        pad_token_id=50258,  # this is out of vocabulary but suppressing warning
        **({'action': actions}),
    )

    recon_output = tokenizer.detokenize(generated_tokens, args.context_length)
    recon_output = recon_output.clamp(0.0, 1.0)
    print(recon_output.shape)

    return recon_output[-1]



def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type = str, default = None)
    parser.add_argument("--trm", type = str, required=True)
    parser.add_argument("--tkn", type = str, required= True)
    parser.add_argument("--expname", type = str, default = "debug")
    parser.add_argument("--traj", type = str, required = True)
    parser.add_argument("--mode", type = str, default = "gt")
    parser.add_argument("--seglength", type = int, default = 12)

    args = parser.parse_args()

    transformer, tokenizer = load_world_model(args)
    
    npz_parser = NPZParser(args.seglength, 250)
    gt_frames, actions = npz_parser.parse(args.traj, "tfds_robonet", load_action=True)



    if args.policy is None:
        print("Running", ("dummy" if args.mode == dummy else "ground truth") ,"actions")

    else:
        policy = load_policy(args)
        print(f"Running with behavior cloning policy.")
    
    # world model gym
    gym = WorldModelEnv(tokenizer, transformer, gt_frames[0], args.seglength, ctx_len)

    all_frames = []

    # run prediction here
    dummy = torch.ones((5)) 
    for i in range(args.seglength):

        processed_frame = torch.nn.functional.pad(gym.frames[-1], (5,5,5,5), mode = "constant", value = 0)
        if policy is not None:
            action = policy(processed_frame)
        elif mode == "gt":
            action = actions[i]
        elif mode == "dummy":
            action = dummy
        next_frame = gym.step(action, predict_step) 
        all_frames.append(next_frame)




    # save images
    recon_output = gym.frames # what the fuck? check that this works?
    frames = [np.concatenate([gt_frames[i], recon_frames[i]], axis=1) for i in range(len(gt_frames))]
    imageio.mimsave(f"{save_path}/{args.expname}_{1}.gif", frames, fps=4, loop=0)
        

if __name__ == "__main__":
    main()

