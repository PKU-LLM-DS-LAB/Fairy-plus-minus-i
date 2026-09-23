import os
import time
from datasets import load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_utils import get_last_checkpoint
from model.modeling_Fairy_plus_minus_i import ComplexNetLM
from model.configuration_Fairy_plus_minus_i import ComplexNetConfig
from transformers.trainer_utils import total_processes_number
from transformers.trainer_callback import TrainerCallback
import itertools
import torch
from transformers import set_seed
from accelerate import Accelerator
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True,
                    help="Path to dataset")
parser.add_argument("--report_to", type=str, default="none",
                    help="Logging integration passed to TrainingArguments, e.g. none, swanlab, tensorboard")
parser.add_argument("--swanlab_workspace", type=str, default=None,
                    help="SwanLab workspace (organization); defaults to your personal workspace")
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

total_steps = 1  

EOS_TOKEN_ID = tokenizer.eos_token_id
BOS_TOKEN_ID = tokenizer.bos_token_id
PAD_TOKEN_ID = tokenizer.pad_token_id
BLOCK_SIZE = 2048

def get_custom_lr_lambda(
    total,
    stage_boundary_ratio=0.5,
    first_stage_scale=1,
    second_stage_scale=0.666,
    warmup=500,
):
    def lr_lambda(current_step: int):
        progress = current_step / total
        if progress < stage_boundary_ratio:
            if current_step < warmup:
                return first_stage_scale * (current_step / warmup)
            else:
                return first_stage_scale * (1 - (current_step - warmup) / total)
        else:
            return second_stage_scale * (1 - progress)

    return lr_lambda


accelerator = Accelerator()
if accelerator.is_main_process:
    print("Accelerator is initialized successfully.")

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

ComplexNetLM.register_for_auto_class("AutoModelForCausalLM")
ComplexNetConfig.register_for_auto_class()


GLOBAL_BATCH_SIZE = 512
PER_DEVICE_BS = 4
MAX_TOKEN = 104_857_600_000
TRAIN_NAME = "out"
OUTPUT_DIR = f"./{TRAIN_NAME}/results"
OUTPUT_MODEL_DIR = f"./{TRAIN_NAME}/saved_model"
LOGGING_DIR = f"./{TRAIN_NAME}/logs"
RESUME = True
MAX_STEPS = MAX_TOKEN // (GLOBAL_BATCH_SIZE * 2048)


new_vocab_size = len(tokenizer)
print(
    f"New vocab size: {new_vocab_size},eos{EOS_TOKEN_ID},bos{BOS_TOKEN_ID},pad{PAD_TOKEN_ID}"
)
train_dataset = load_from_disk(args.dataset_path)
train_dataset = train_dataset.shuffle(seed=42)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


config = ComplexNetConfig(
    vocab_size=new_vocab_size,
    attn_implementation="flash_attention_2",  # Set to "flash_attention_2" for flash attention
)

model = ComplexNetLM(config)
print(config._attn_implementation)


training_args = TrainingArguments(
    max_steps=MAX_STEPS,
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=PER_DEVICE_BS,
    learning_rate=1.5e-3,
    max_grad_norm=1.0,
    warmup_steps=375,
    weight_decay=0.1,
    logging_dir=LOGGING_DIR,
    save_strategy="steps",
    save_steps=1000,
    bf16=True,
    adam_beta1=0.9,
    adam_beta2=0.95,
    report_to=args.report_to,
    logging_steps=10,
    save_total_limit=100,
)

world_size = total_processes_number(training_args.local_rank)
assert GLOBAL_BATCH_SIZE % (world_size * PER_DEVICE_BS) == 0, (
    f"Global batch size {GLOBAL_BATCH_SIZE} must be divisible by "
    f"({world_size} * {PER_DEVICE_BS}) = {world_size * PER_DEVICE_BS}"
)
accumulation_steps = GLOBAL_BATCH_SIZE // (world_size * PER_DEVICE_BS)
training_args.gradient_accumulation_steps = accumulation_steps
if accelerator.is_main_process:
    print(
        f"global batch size: {GLOBAL_BATCH_SIZE}, per device batch size: {training_args.per_device_train_batch_size}, "
        f"gradient accumulation steps: {training_args.gradient_accumulation_steps}, "
    )
    print(f"total parameters: {model.num_parameters() / 1e6:.2f} M")
    print(train_dataset)


callbacks = []
use_swanlab = accelerator.is_main_process and args.report_to == "swanlab"
if use_swanlab:
    import swanlab as swanlab
    from swanlab.integration.transformers import SwanLabCallback
    from swanlab import Settings
    cfg = training_args.to_dict()
    cfg["model_type"] = "ComplexNetLM"
    cfg["dataset"] = "RedPajama-Data-1T-100B"
    cfg["max_length"] = 2048  
    swanlab.init(
        workspace=args.swanlab_workspace,
        project="complexnet-training",
        name=time.strftime("%m%d%H%M%S") + "redpajama_100B",
        config=cfg,
        settings=Settings(
            requirements_collect=False,
        ),
 
    )
    callbacks.append(SwanLabCallback())


class CustomWeightDecayCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        if optimizer is not None:
           
            total_step = state.max_steps
            current_step = state.global_step

            progress = current_step / total_step
            weight_decay = 0.1 if progress < 0.5 else 0.0
            for group in optimizer.param_groups:
                group["weight_decay"] = weight_decay
        else:
            print("Optimizer None")


callbacks.append(CustomWeightDecayCallback())


class CustomTrainer(Trainer):
    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        global total_steps
        total_steps = num_training_steps
        if optimizer is None:
            optimizer = self.optimizer

        lr_lambda = get_custom_lr_lambda(
            total=num_training_steps,
            stage_boundary_ratio=0.5,
            first_stage_scale=1,
            second_stage_scale=0.6666,
            warmup=training_args.warmup_steps,
        )

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        self.lr_scheduler = scheduler
        self._created_lr_scheduler = True
        return self.lr_scheduler


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=data_collator,
    callbacks=callbacks,
)


if not os.path.isdir(training_args.output_dir):
    try:
        os.makedirs(training_args.output_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError("Can't create directory: " + training_args.output_dir) from e
last_ckpt = get_last_checkpoint(training_args.output_dir)
try:
    if not RESUME:
        last_ckpt = None
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

finally:
    if use_swanlab:
        swanlab.finish()
