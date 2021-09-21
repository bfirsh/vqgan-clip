import sys

from model import (
    load_vqgan_model,
    parse_prompt,
    MakeCutouts,
    Prompt,
    resize_image,
    vector_quantize,
    clamp_with_grad,
)


import clip
import cog
import pathlib
from PIL import Image
import torch
from torchvision import transforms
from torch.nn import functional as F
from torchvision.transforms import functional as TF


class VQGANCLIP(cog.Predictor):
    def setup(self):
        clip_model = "ViT-B/32"
        vqgan_config = "vqgan_imagenet_f16_1024.yaml"
        vqgan_checkpoint = "vqgan_imagenet_f16_1024.ckpt"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(self.device)
        self.perceptor = (
            clip.load(clip_model, jit=False)[0]
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )

    @cog.input("prompt", type=str, help="Text prompt")
    @cog.input("iterations", type=int, help="Number of iterations", default=500)
    @cog.input("display_freq", type=int, default=50, help="Display frequency")
    def predict(self, prompt, iterations, display_freq):
        prompts = [prompt]
        image_prompts = []
        noise_prompt_seeds = ([],)
        noise_prompt_weights = []
        size = [480, 480]
        init_image = None
        init_weight = 0.0
        step_size = 0.05
        cutn = 64
        cut_pow = 1.0
        seed = 0

        cut_size = self.perceptor.visual.input_resolution
        e_dim = self.model.quantize.e_dim
        f = 2 ** (self.model.decoder.num_resolutions - 1)
        make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=cut_pow)
        n_toks = self.model.quantize.n_e
        toksX, toksY = size[0] // f, size[1] // f
        sideX, sideY = toksX * f, toksY * f
        z_min = self.model.quantize.embedding.weight.min(dim=0).values[
            None, :, None, None
        ]
        z_max = self.model.quantize.embedding.weight.max(dim=0).values[
            None, :, None, None
        ]

        if seed is not None:
            torch.manual_seed(seed)

        if init_image:
            pil_image = Image.open(init_image).convert("RGB")
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            z, *_ = self.model.encode(
                TF.to_tensor(pil_image).to(self.device).unsqueeze(0) * 2 - 1
            )
        else:
            one_hot = F.one_hot(
                torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks
            ).float()
            z = one_hot @ self.model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        z_orig = z.clone()
        z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=step_size)

        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        pMs = []

        for prompt in prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.perceptor.encode_text(
                clip.tokenize(txt).to(self.device)
            ).float()
            pMs.append(Prompt(embed, weight, stop).to(self.device))

        for prompt in image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(path).convert("RGB"), (sideX, sideY))
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
            embed = self.perceptor.encode_image(normalize(batch)).float()
            pMs.append(Prompt(embed, weight, stop).to(self.device))

        for seed, weight in zip(noise_prompt_seeds, noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(
                generator=gen
            )
            pMs.append(Prompt(embed, weight).to(self.device))

        def synth(z):
            z_q = vector_quantize(
                z.movedim(1, 3), self.model.quantize.embedding.weight
            ).movedim(3, 1)
            return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

        @torch.no_grad()
        def checkin(i, losses):
            losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
            sys.stderr.write(f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}\n")
            out = synth(z)
            TF.to_pil_image(out[0].cpu()).save("progress.png")
            # display.display(display.Image('progress.png'))

        def ascend_txt():
            out = synth(z)
            iii = self.perceptor.encode_image(normalize(make_cutouts(out))).float()

            result = []

            if init_weight:
                result.append(F.mse_loss(z, z_orig) * init_weight / 2)

            for prompt in pMs:
                result.append(prompt(iii))

            return result

        def train(i):
            opt.zero_grad()
            lossAll = ascend_txt()
            if i % display_freq == 0:
                checkin(i, lossAll)
            loss = sum(lossAll)
            loss.backward()
            opt.step()
            with torch.no_grad():
                z.copy_(z.maximum(z_min).minimum(z_max))

        for i in range(iterations):
            train(i)

        return pathlib.Path("progress.png")
