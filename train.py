import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import argparse
from utils.dataset import ContentDataset
from utils.losses import LossNetwork, style_loss, content_loss
from models.video_transformer import CompoundTransformerNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#training function
def train_style_transfer(
        content_dir, #directory containing content images
        style_img, #path to style image
        epochs=2,
        batch_size=4, 
        lr=1e-3,
        train_img_size=256,
        content_weight=1,
        style_weight=1e6,
        save_path='style_transfer_net.pt',
        device=device,
):
    #1. Setup transforms, dataset, dataloader
    train_transform = T.Compose([
        T.ToPILImage(), #ensures image is converted to PIL image
        T.Resize(train_img_size, T.InterpolationMode.BICUBIC),
        T.CenterCrop(train_img_size), #if varying aspect ratios, will crop to square 
        T.ToTensor(),
    ])
    dataset = ContentDataset(content_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    #2. Load style image
    style_img = T.ToTensor()(
        T.Resize(train_img_size, T.InterpolationMode.BICUBIC)
                (Image.open(style_img).convert('RGB')))
    style_img = style_img.unsqueeze(0).to(device) #add batch dimension, shape (1, 3, H, W)
    style_img = style_img.repeat(batch_size, 1, 1, 1) #repeat style image to match batch size

    #3. Initialize model and move it to device
    model = CompoundTransformerNet(
        low_layer=5, high_layer=19, #conv2_1 and conv4_1 in VGG19
        c_channels_low=128, c_channels_high=512,
        s_channels_low=128, s_channels_high=512,
        mid_channels=64
    ).to(device)

    #4. Initialize loss network
    loss_net = LossNetwork(content_layer=21, style_layers=(0, 5, 10, 19, 28)).to(device)
    loss_net.eval() #freeze params

    #5. Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #We store style features once (as single style image)
    with torch.inference_mode():
        style_feats = loss_net(style_img)

    #Training loop
    for epoch in range(epochs):
        for i, content_batch in enumerate(dataloader):
            content_batch = content_batch.to(device)

            #1. Forward pass: stylize
            stylized_batch = model(content_batch, style_img)

            #2. Compute losses
            stylized_feats = loss_net(stylized_batch)
            content_feats = loss_net(content_batch)

            #2.1 Compute content loss (e.g loss at content layer (21/conv4_2 in VGG19))
            c_loss = content_loss(stylized_feats['content'], content_feats['content'])

            #2.2 Compute style loss (e.g loss at style layers (0, 5, 10, 19, 28 in VGG19))
            s_loss = 0.0
            for idx in (0, 5, 10, 19, 28):
                if idx in stylized_feats:
                    s_loss += style_loss(stylized_feats[idx], style_feats[idx])

            #2.3 Compute total loss
            total_loss = (content_weight * c_loss) + (style_weight * s_loss)

            #3. Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            #4. Logging
            if i % 100 == 0:
                print(f'''Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], 
                      Content Loss: {c_loss.item():.4f}, 
                      Style Loss: {s_loss.item():.6f},
                      Total Loss: {total_loss.item():.4f}''')
            
    #Save the trained model
    torch.save(model.state_dict(), save_path)

    print(f"Training Complete. Model saved at {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for fast style transfer")
    parser.add_argument('--content_dir', type=str, required=True, help="Directory containing content images")
    parser.add_argument('--style_img', type=str, required=True, help="Path to style image")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--train_img_size', type=int, default=256, help="Training image size")
    parser.add_argument('--content_weight', type=float, default=1, help="Content loss weight")
    parser.add_argument('--style_weight', type=float, default=1e6, help="Style loss weight")
    parser.add_argument('--save_path', type=str, default='style_transfer_net.pt', help="Path to save trained model (.pt)")
    args = parser.parse_args()

    train_style_transfer(
        content_dir=args.content_dir,
        style_img=args.style_img,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_img_size=args.train_img_size,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        save_path=args.save_path,
        device=device,
    )