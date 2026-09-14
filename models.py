import torch
import torch.nn as nn

class AdvancedUNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_fonts=5,
        use_skip=True,
        use_vae=False,
        compress_bottleneck=False
    ):
        super(AdvancedUNetGenerator, self).__init__()
        self.use_skip = use_skip
        self.use_vae = use_vae
        self.compress_bottleneck = compress_bottleneck and (not use_skip) and (not use_vae)
        
        # Font category embedding (integrates labels into the generation process)
        self.category_emb = nn.Embedding(num_fonts, 128)
        self.emb_fc = nn.Linear(128, 256 * 256)

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = self._make_enc_block(64, 128)
        self.enc3 = self._make_enc_block(128, 256)
        self.enc4 = self._make_enc_block(256, 512)
        
        # Core VAE logic (Factor III requirement)
        if self.use_vae:
            self.fc_mu = nn.Linear(512 * 16 * 16, 512)
            self.fc_logvar = nn.Linear(512 * 16 * 16, 512)
            self.fc_decode = nn.Linear(512, 512 * 16 * 16)
        elif self.compress_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # Decoder
        self.dec4 = self._make_dec_block(512 if not use_skip else 512*2, 256)
        self.dec3 = self._make_dec_block(256 if not use_skip else 256*2, 128)
        self.dec2 = self._make_dec_block(128 if not use_skip else 128*2, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 if not use_skip else 64*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _make_enc_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _make_dec_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        # Inject condition labels
        emb = self.category_emb(labels)
        emb_img = self.emb_fc(emb).view(-1, 1, 256, 256)
        x = torch.cat([x, emb_img], dim=1)

        # Encoder forward pass
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        kl_loss = None
        # VAE or Bottleneck (Factor II & III logic)
        if self.use_vae:
            flat_e4 = e4.view(e4.size(0), -1)
            mu = self.fc_mu(flat_e4)
            logvar = self.fc_logvar(flat_e4)
            # Apply clamp to prevent logvar explosion
            logvar = torch.clamp(logvar, min=-10, max=10)
            z = self.reparameterize(mu, logvar)
            b = self.fc_decode(z).view(e4.size())
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            b = self.bottleneck(e4)

        # Decoder forward pass (with Skip-Connection toggle)
        if self.use_skip:
            d4 = self.dec4(torch.cat([b, e4], dim=1))
            d3 = self.dec3(torch.cat([d4, e3], dim=1))
            d2 = self.dec2(torch.cat([d3, e2], dim=1))
            out = self.dec1(torch.cat([d2, e1], dim=1))
        else:
            d4 = self.dec4(b)
            d3 = self.dec3(d4)
            d2 = self.dec2(d3)
            out = self.dec1(d2)

        return out, kl_loss

# WGAN-GP Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    def forward(self, source, target):
        return self.model(torch.cat([source, target], dim=1))

# WGAN-GP Gradient Penalty Calculation
def compute_wgan_gp_loss(discriminator, real_imgs, fake_imgs, source_imgs, device):
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
    d_interpolates = discriminator(source_imgs, interpolates)
    
    # FIX: Use ones_like to dynamically match d_interpolates shape (e.g., 30x30), preventing hardcoded size errors
    fake = torch.ones_like(d_interpolates).to(device)
    
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
