import torch
import argparse

from torch.utils.data import DataLoader
from ivideogpt.data import *
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
state_dict_path = "" # TODO replace with correct path please.

class DinoPolicy(torch.nn.Module):
    
    def __init__(self):
        super(DinoPolicy,self).__init__()
        
        self.tokenizer = dinov2_vits14 # TODO what size are the tokens this outputs



        self.actor = torch.nn.Sequential(
                    torch.nn.Linear(384, 2048),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2048, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 5),
                    torch.nn.ReLU()
        ) # 5 dim action space so 5 dim outputs

    def forward(self, x):

        self.tokenizer.eval()

        tokens = self.tokenizer(x) # is there any preprocessing I need to do on the image prior? idts
        
        output = self.actor(tokens)

        return output


def make_loaders():
    
    
    batch_size = 16

    augmentation_args = {
            'brightness': [0.9, 1.1],
            'contrast': [0.9, 1.1],
            'saturation':[0.9,1.1],
            'hue':[-0.05, 0.05],
            'random_resized_crop_scale':(0.8,1.0),
            'random_resized_crop_ratio':(0.9,1.1),
            'no_aug':False,
    }
    
    segment_args = {
        'random_selection': False,
        'random_shuffle': False,
        'goal_conditioned': False,
        'segment_length': 12,
        'context_length': 2,
        'stepsize': 1, # todo, idk what this should be?
        'segment_horizon': None,
    }

    train_loader = SimpleRoboticDataLoaderv2(
                parent_dir = "/data2/frame_datasets",
                datasets = DATASET_NAMED_MIXES['tfds_robonet'],
                batch_size = batch_size,
                num_workers = 4,
                train = True,
                maxsize = None,
                image_size = 256,
                sthsth_root_path = None,
                **augmentation_args,
                **segment_args,
                load_action = True
            )
    
    val_loader = SimpleRoboticDataLoaderv2(
                parent_dir = "/data2/frame_datsets",
                datasets = DATASET_NAMED_MIXES['tfds_robonet'],
                batch_size = batch_size,
                num_workers = 4,
                train = True,
                maxsize = None,
                image_size = 256,
                sthsth_root_path = None,
                **augmentation_args,
                **segment_args,
                load_action = True
            ) # always load actions for our purposes


    return train_loader, val_loader 


def load_world_model():
     # we use args directly here for simplicity's sake
    # LOAD TOKENIZER
    tokenizer = CompressiveVQModel.from_pretrained(
        tokenizer_path,
        subfolder=None,
        low_cpu_mem_usage=False).to(device)
    tokenizer.eval()


    # LOAD TRANSFORMER
    perlude_tokens_num = (256) # TODO check the actual calculation for these numbers
    tokens_per_dyna = 16

    state_dict_path = transformer_path
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
                                    segment_length=12, model_type='llama',
                                    reward_prediction=False, action_recon=None)
    transformer.load_state_dict(state_dict, strict=True)
    transformer.eval()
    transformer.to(device)

    return transformer, tokenizer


def train_model(train_loader, val_loader, model_pth = None):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

   
    batch_size = 32
    epochs = 1
    model = DinoPolicy().to(device)
    skip_batches = 0
    if model_pth is not None:
        skip_batches = int(model_pth[model_pth.find("ep")+2:model_pth.find("_l")])
        print("loading model from ckpt")
        print(f"skipping {skip_batches} batches") 
        model.load_state_dict(torch.load(model_pth))

    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        batch_it = tqdm(enumerate(train_loader), desc="batch", unit="batches", total = len(train_loader))
        running_loss = 0.0
        for batch_id, inputs in batch_it:
            if(batch_id <= skip_batches):
                continue


            # do the roar
            pixel_values, actions = inputs
            

            # pixel values has shape (B, T, C, H, W), some permute of C,H,W
            # flatten it along the T dimension. It should become (B*T, C, H, W)
            # Same with actions
            pixel_values = torch.flatten(pixel_values, end_dim = 1)
            actions = torch.flatten(actions, end_dim = 1)

            pixel_values = torch.nn.functional.pad(pixel_values, (5,5,5,5), mode = "constant", value = 0)
            # print("pv shape:",pixel_values.shape)
            # print("act shape:",actions.shape)

            actions = actions.to(device)
            pixel_values = pixel_values.to(device)

            # check the shape of pixel_values being passed in. we want a massive set of images and actions. 
            outputs = model(pixel_values)
            optimizer.zero_grad()
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            

            running_loss += loss.item()

            batch_it.set_postfix({"current loss" : loss.item()})
            if (batch_id+1) % 1000 == 0:
                torch.save(model.state_dict(), f"dino_policy_ep{batch_id+1}_l{loss.item()}.pth")
        
        print(f"Average loss: {running_loss}")
                
    print("done with simple train")

    torch.save(model.state_dict(), "dino_policy.pth")






def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type = str, default = None)

    args = parser.parse_args()
    

    train_loader, val_loader = make_loaders()

    train_model(train_loader, val_loader, args.ckpt)


    # etc etc

if __name__ == "__main__":
    main()
