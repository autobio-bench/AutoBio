import logging

import numpy as np
import torch

from scripts.autobio_model import AutoBioRDT
from autobio_inference.base_policy import BasePolicy
from autobio_inference.websocket_policy_server import WebsocketPolicyServer

class Policy(BasePolicy):
    def __init__(self, checkpoint_path):
        self.model = AutoBioRDT(checkpoint_path)

    @torch.inference_mode()
    def infer(self, obs: dict | list[dict]):
        if isinstance(obs, dict):
            obs = [obs]
            single = True
        else:
            single = False
        proprio = np.stack([o["observation/state"] for o in obs], axis=0)
        proprio = proprio[:, None, :]

        images = []
        for o in obs:
            images += [
                o["observation/-1/image"],
                o["observation/-1/wrist_image"],
                o["observation/-1/wrist_image_2"],
                o["observation/image"],
                o["observation/wrist_image"],
                o["observation/wrist_image_2"],
            ]
        image_embeds = self.model.encode_image(images)
        image_embeds = image_embeds.reshape(len(obs), 6 * image_embeds.shape[1], *image_embeds.shape[2:])

        prompt = [o["prompt"] for o in obs]
        text_embeds, text_mask = self.model.get_instruction_embeds(prompt)

        trajectory = self.model.step(proprio, image_embeds, text_embeds, text_mask)
        trajectory = trajectory.cpu().numpy()
        if single:
            return {"actions": trajectory[0]}
        else:
            return [{"actions": t} for t in trajectory]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Serve a policy using the websocket protocol.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to.")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    policy = Policy(args.checkpoint_path)
    server = WebsocketPolicyServer(policy, args.host, args.port)
    server.serve_forever()
