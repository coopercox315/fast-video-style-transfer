import argparse
import os
import torch
import cv2
import torchvision.transforms as T
from PIL import Image
import numpy as np
from models.video_transformer import CompoundTransformerNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_video_file(filename):
    """
    Simple function to check if a file is a video file.
    Returns:
    - True if file extension is typical for video.
    """
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')

    return filename.lower().endswith(video_exts)

def is_image_file(filename):
    """
    Simple function to check if a file is an image file.
    Returns:
    - True if file extension is typical for images.
    """
    img_exts = ('.jpg', '.jpeg', '.png')

    return filename.lower().endswith(img_exts)

def load_image(img_path, img_size=512):
    """
    Load an image from a given path, resize and convert to tensor.
    """
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)

    return img

def style_transfer_image(model, content_tensor, style_tensor, device):
    """
    Applies style transfer to a single image (tensor).
    Returns:
    - Tensor: the stylized image tensor [1, C, H, W].
    """
    model.to(device).eval()
    with torch.inference_mode():
        content_tensor = content_tensor.to(device)
        style_tensor = style_tensor.to(device)
        output = model(content_tensor, style_tensor)

    return output.cpu() #return tensor to CPU for processing/saving

def style_transfer_video(model, content_video_path, style_tensor, output_video_path='styled_output.mp4', size=512, device=device):
    """
    Applies style transfer to every frame of a video.
    Writes stylized output frames to a new video file and saves to output_video_path.
    """
    cap = cv2.VideoCapture(content_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {content_video_path}")
    
    #get original FPS, size of video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #define output video size
    out_w, out_h = width, height
    if size is not None:
        max_dim = max(width, height)
        if max_dim > size:
            scale = size / float(max_dim)
            out_w = int(width * scale)
            out_h = int(height * scale)

    #initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

    #preprocess the style only once (as it is the same for all frames)
    model.to(device).eval()
    with torch.inference_mode():
        style_tensor = style_tensor.to(device)

        #read and process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            #1. Convert frame (BGR) -> RGB PIL -> tensor
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            transform = T.Compose([
                T.Resize((out_h, out_w)),
                T.ToTensor(),
            ])
            frame_tensor = transform(pil_frame).unsqueeze(0).to(device)

            #2. Apply style transfer to frame
            stylized_frame_tensor = model(frame_tensor, style_tensor) #shape [1, 3, out_h, out_w]

            #3. Convert tensor -> numpy -> BGR
            stylized_frame_tensor = stylized_frame_tensor.squeeze(0).cpu()
            stylized_frame_np = stylized_frame_tensor.permute(1, 2, 0).numpy()
            stylized_frame_np = (stylized_frame_np * 255).clip(0, 255).astype(np.uint8)
            stylized_frame_bgr = cv2.cvtColor(stylized_frame_np, cv2.COLOR_RGB2BGR)

            #4. Write frame to output video
            out.write(stylized_frame_bgr)

    #release resources
    cap.release()
    out.release()
    print(f"Stylized video saved to {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Feed-forward NST for images or videos.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to saved model (.pt or .pth)")
    parser.add_argument('--content', type=str, required=True, help="Path to content image or video")
    parser.add_argument('--style', type=str, required=True, help="Path to style image")
    parser.add_argument('--output', type=str, default='styled_output', help="Name of output file")
    parser.add_argument('--size', type=int, default=512, help="Output content size (max dimension)")
    args = parser.parse_args()

    #1. Load model
    model = CompoundTransformerNet()
    model.load_state_dict(torch.load(args.model_path), maps_location=device)

    #2. Load style image
    style_img = load_image(args.style, args.size) #shape [1, 3, H, W]

    #3. Check if content is image or video
    if is_image_file(args.content):
        #add .jpg extension to output_path if not present
        output_path = args.output
        if not args.content.lower().endswith(('.jpg', '.jpeg', '.png')):
            output_path += '.jpg'

        #single image inference
        content_img = load_image(args.content, args.size) #shape [1, 3, H, W]
        styled_img = style_transfer_image(model, content_img, style_img, device)

        #convert for saving
        out_img = styled_img.squeeze(0).permute(1, 2, 0).numpy()
        out_img = (out_img * 255).clip(0, 255).astype(np.uint8)
        out_img = Image.fromarray(out_img)
        out_img.save(output_path)
        print(f"Stylized image saved to {output_path}")

    elif is_video_file(args.content):
        #add .mp4 extension to output_path if not present
        output_path = args.output
        if not args.content.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path += '.mp4'
        
        #video inference
        style_transfer_video(model, args.content, style_img, output_path, args.size, device)

    else:
        raise ValueError("Content file must be a supported image or video format (.jpg, .jpeg, .png, .mp4, .avi, .mov, .mkv)")
    
if __name__ == "__main__":
    main()  