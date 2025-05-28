import os

import numpy as np
import torch
from PIL import Image

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner


AUTOBIO_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
] + [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["left_gripper_open"]
]

class AutoBioRDT(object):
    def __init__(
        self,
        checkpoint_dir,
        *,
        device='cuda',
        dtype=torch.bfloat16,
        control_frequency=50,
        pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl",
        pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
        tokenizer_max_length=48,
        state_token_dim=128,
    ):
        self.dtype = dtype
        self.state_token_dim = state_token_dim
        self.device = device
        self.control_frequency = control_frequency
        # We do not use the text encoder due to limited GPU memory
        self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path, tokenizer_max_length)
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.policy = RDTRunner.from_pretrained(checkpoint_dir)
        
        with open(os.path.join(checkpoint_dir, "normalization.json"), "r") as f:
            import json
            normalization = json.load(f)
            self.state_min = torch.tensor(normalization["state_min"], dtype=self.dtype, device=self.device)
            self.state_max = torch.tensor(normalization["state_max"], dtype=self.dtype, device=self.device)
            self.actions_min = torch.tensor(normalization["actions_min"], dtype=self.dtype, device=self.device)
            self.actions_max = torch.tensor(normalization["actions_max"], dtype=self.dtype, device=self.device)

        self.state_indices = AUTOBIO_STATE_INDICES
        self.actions_indices = AUTOBIO_STATE_INDICES
        if len(self.state_min) == 7:
            self.state_indices = AUTOBIO_STATE_INDICES[:7]
            self.actions_indices = AUTOBIO_STATE_INDICES[:7]

        self.reset()

    def get_text_encoder(self, pretrained_text_encoder_name_or_path, tokenizer_max_length=1024):
        text_embedder = T5Embedder(from_pretrained=pretrained_text_encoder_name_or_path, 
                                   model_max_length=tokenizer_max_length, 
                                   device=self.device)
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        self.text_model.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)

        self.instruction_cache = {}

    def encode_instruction(self, instruction):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device
        
        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        res = self.text_tokenizer(
            instruction, return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        tokens = res["input_ids"].to(self.device)
        attention_mask = res["attention_mask"].to(self.device)

        with torch.no_grad():
            text_embeds = self.text_model(tokens, attention_mask).last_hidden_state.detach()

        return text_embeds, attention_mask
    
    def get_instruction_embeds(self, instructions: list[str]):
        text_embeds = []
        text_mask = []
        for instruction in instructions:
            if instruction in self.instruction_cache:
                embeds, mask = self.instruction_cache[instruction]
            else:
                embeds, mask = self.encode_instruction(instruction)
                embeds = embeds.squeeze(0).cpu()
                mask = mask.squeeze(0).bool().cpu()
                self.instruction_cache[instruction] = (embeds, mask)
            text_embeds.append(embeds)
            text_mask.append(mask)
        text_embeds = torch.stack(text_embeds, axis=0).to(self.device)
        text_mask = torch.stack(text_mask, axis=0).to(self.device)
        return text_embeds, text_mask

    def encode_image(self, images):
        # The background image used for padding
        background_color = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"], 
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        
        # Preprocess the images by order and encode them
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)
                                
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(self.device, dtype=self.dtype)

        image_embeds = self.vision_model(image_tensor).detach()
        return image_embeds

    def _format_state(self, state):
        state_dim = state.shape[-1]
        assert state_dim in (7, 14)
        state = (state - self.state_min) / (self.state_max - self.state_min)
        
        B, N, _ = state.shape
        uni_state = torch.zeros(
            (B, N, self.state_token_dim), 
            device=state.device, dtype=state.dtype
        )
        uni_state[:, :, self.state_indices] = state
        uni_state_elem_mask = torch.zeros(
            (B, self.state_token_dim),
            device=state.device, dtype=state.dtype
        )
        uni_state_elem_mask[:, self.state_indices] = 1
        return uni_state, uni_state_elem_mask

    def _unformat_actions(self, uni_actions):
        actions = uni_actions[:, :, self.actions_indices]
        actions = actions * (self.actions_max - self.actions_min) + self.actions_min
        return actions

    @torch.no_grad()
    def step(self, proprio, image_embeds, text_embeds, text_mask):
        device = self.device
        dtype = self.dtype
    
        # Prepare the proprioception states and the control frequency
        joints = torch.tensor(proprio, device=device, dtype=dtype)
        states, state_elem_mask = self._format_state(joints)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device).expand(len(states))
        
        text_embeds = text_embeds.to(device, dtype=dtype)
        
        # Predict the next action chunk given the inputs
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=text_mask,
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),  
            ctrl_freqs=ctrl_freqs
        )
        trajectory = self._unformat_actions(trajectory).to(torch.float32)

        return trajectory
