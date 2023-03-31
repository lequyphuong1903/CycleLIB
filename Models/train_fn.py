import torch
from tqdm import tqdm
from torchvision.utils import save_image
import os
def train(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, param, epoch
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (A, B) in enumerate(loop):
        A = A.to(param.device)
        B = B.to(param.device)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_A = gen_H(B)
            D_H_real = disc_H(A)
            D_H_fake = disc_H(fake_A.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_B = gen_Z(A)
            D_Z_real = disc_Z(B)
            D_Z_fake = disc_Z(fake_B.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_A)
            D_Z_fake = disc_Z(fake_B)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_B = gen_Z(fake_A)
            cycle_A = gen_H(fake_B)
            cycle_B_loss = l1(B, cycle_B)
            cycle_A_loss = l1(A, cycle_A)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_B = gen_Z(B)
            identity_A = gen_H(A)
            identity_B_loss = l1(B, identity_B)
            identity_A_loss = l1(A, identity_A)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_B_loss * param.lambda_cy
                + cycle_A_loss * param.lambda_cy
                + identity_A_loss * param.lambda_id
                + identity_B_loss * param.lambda_id
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if os.path.exists("saved_images"):
            if idx % 200 == 0:
                save_image(fake_A * 0.5 + 0.5, f"saved_images/A_{epoch}_{idx}.png")
                save_image(fake_B * 0.5 + 0.5, f"saved_images/B_{epoch}_{idx}.png")
        else:
            os.mkdir("saved_images")
            
        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))