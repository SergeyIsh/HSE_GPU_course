import argparse
import numpy as np
import aiohttp
import asyncio
import os
from PIL import Image
import io

async def send_to_server(session, server_url, image_binary):
    async with session.post(server_url + "/sobel", data=image_binary) as response:
        if response.status != 200:
            error = await response.text()
            print(f"Error from server {server_url}: {error}")
            return None
        return await response.json()

async def process_image(session, server_url, image_path, output_dir):
    image = Image.open(image_path)
    image = np.array(image)
    h, w, _ = image.shape
    img_pil = Image.fromarray(image)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    image_binary = buffer.getvalue()

    response = await send_to_server(session, server_url, image_binary)
    if response is not None:
        image_borders = np.array(response["image_borders"], dtype=np.uint8).reshape((h, w))
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        Image.fromarray(image_borders).save(output_path)
        print(f"Result image is savet to: {output_path}")
        return response.get("total_gpu_time_ms", 0)
    return 0

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servers", nargs="+", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image_files = [os.path.join(args.images, f) for f in os.listdir(args.images) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in directory: {args.images}")
        return

    start_time = asyncio.get_event_loop().time()
    total_gpu_time_ms = 0
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for image_path in image_files:
            server_url = args.servers[len(tasks) % len(args.servers)]
            task = asyncio.create_task(process_image(session, server_url, image_path, args.output))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_gpu_time_ms = sum(results)
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    print(f"All images processed. Total time: {total_time:.6f} seconds")
    print(f"Total GPU time: {total_gpu_time_ms:.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())
