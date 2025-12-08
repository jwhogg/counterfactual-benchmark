#imports from root to use local packages
import os, sys
from models.gans.celeba_gan import CelebaCondGAN
from datasets.celeba.dataset import Celeba
from models.classifiers.celeba_classifier import CelebaClassifier
import torch
import joblib
from pytorch_lightning import Trainer
from torchvision.transforms import RandomHorizontalFlip
from datasets.transforms import SelectParentAttributesTransform
from models.utils import generate_checkpoint_callback, generate_early_stopping_callback, generate_ema_callback

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

def get_dataloaders(data_class, attribute_size, config, transform=None, **kwargs):
    data = data_class(data_dir="datasets\celeba\data", attribute_size=attribute_size, transform=transform, split='train', **kwargs)

    if data.has_valid_set:
        train_set = data
        val_set = data_class(data_dir="datasets\celeba\data", attribute_size=attribute_size, transform=transform, split='valid', **kwargs)
    else:
        train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])

    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, num_workers=7)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False, num_workers=7)
    return train_data_loader, val_data_loader

def train_gan(gan, config, data_class, graph_structure, attribute_size, checkpoint_dir, **kwargs):
    transform = SelectParentAttributesTransform("image", attribute_size, graph_structure)

    train_data_loader, val_data_loader = get_dataloaders(data_class, attribute_size, config, transform, **kwargs)

    monitor= "fid" if config['finetune'] == 0 else "lpips"
    callbacks = [
        generate_checkpoint_callback(gan.name, checkpoint_dir, monitor=monitor),
        generate_early_stopping_callback(patience=config["patience"], monitor=monitor)
    ]


    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto",
                      callbacks=callbacks,
                      default_root_dir=checkpoint_dir, max_epochs=config["max_epochs"])

    trainer.fit(gan, train_data_loader, val_data_loader)

config_cls = {
    "attribute_size": {
        "Smiling": 1,
        "Eyeglasses": 1
    },

    "dataset": "celeba",
    "ckpt_path" : "../methods/deepscm/checkpoints/celeba/simple/trained_classifiers", #modified this line for the notebook
    "in_shape" : [3, 64, 64] ,
    "patience" : 10,
    "batch_size_train" : 128,
    "batch_size_val" : 128,
    "lr" : 1e-3,
    "max_epochs" : 1000,
    "ema": "True"
}

# define causal graph (canibalised from config/../gan.json)
causal_graph = {
        "Smiling": [],
        "Eyeglasses": [],
        "image": ["Smiling", "Eyeglasses"],
    }

#define the models for each mechanism (only one for image here as the rest in the graph are roots)
mechanism_models =  {
        "image": {
            "model_type": "gan",
            "model_class": "CelebaCondGAN",
            "module": "models.gans",
            "params": {
                "n_chan_enc": [3, 64, 128, 256, 256, 512, 512],
                "n_chan_gen": [512, 512, 256, 256, 128, 64, 3],
                "latent_dim": 512,
                "num_continuous": 2,
                "d_updates_per_g_update": 1,
                "gradient_clip_val": 0.5,
                "finetune": 1,
                "pretrained_path": "",
                "lr": 1e-4,
                "batch_size_train": 128,
                "batch_size_val": 128,
                "patience": 10,
                "max_epochs": 1000
            }
        }
    }

attribute_size = {
        "Smiling": 1,
        "Eyeglasses": 1
         }

if __name__ == "__main__":
    print("Begining training process")

    for variable in causal_graph:
        if variable not in mechanism_models: continue #only want to train variables with models, root variables don't have a causal mechanism
        print("training...")
        train_gan(
            gan=CelebaCondGAN(params=mechanism_models[variable]["params"], attr_size=config_cls["attribute_size"]),
            config=mechanism_models[variable]["params"],
            data_class=Celeba,
            graph_structure=causal_graph,
            attribute_size=attribute_size,
            checkpoint_dir='methods\deepscm\checkpoints\celeba\simple/trained_scm' #adjusted default path because the notebook is down one
            )

    print('done!')