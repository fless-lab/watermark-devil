import asyncio
import aiohttp
import json
from pathlib import Path

async def detect_watermarks(image_path: str, api_key: str):
    """Example of using the watermark detection API"""
    async with aiohttp.ClientSession() as session:
        # Prepare the multipart form data
        data = aiohttp.FormData()
        data.add_field('file',
                      open(image_path, 'rb'),
                      filename=Path(image_path).name,
                      content_type='image/jpeg')
        
        data.add_field('options',
                      json.dumps({
                          'detect_multiple': True,
                          'min_confidence': 0.5,
                          'detection_type': 'all'
                      }))
        
        # Send request to detection endpoint
        headers = {'X-API-Key': api_key}
        async with session.post('http://localhost:8000/detect',
                              data=data,
                              headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error = await response.text()
                raise Exception(f"Detection failed: {error}")

async def remove_watermarks(image_path: str, watermarks: list, api_key: str):
    """Example of using the watermark removal API"""
    async with aiohttp.ClientSession() as session:
        # Prepare the multipart form data
        data = aiohttp.FormData()
        data.add_field('file',
                      open(image_path, 'rb'),
                      filename=Path(image_path).name,
                      content_type='image/jpeg')
        
        data.add_field('watermarks',
                      json.dumps(watermarks))
        
        data.add_field('options',
                      json.dumps({
                          'quality': 'high',
                          'method': 'hybrid',
                          'preserve_details': True
                      }))
        
        # Send request to removal endpoint
        headers = {'X-API-Key': api_key}
        async with session.post('http://localhost:8000/remove',
                              data=data,
                              headers=headers) as response:
            if response.status == 200:
                # Save the cleaned image
                output_path = Path(image_path).with_stem(f"{Path(image_path).stem}_clean")
                with open(output_path, 'wb') as f:
                    f.write(await response.read())
                return output_path
            else:
                error = await response.text()
                raise Exception(f"Removal failed: {error}")

async def main():
    # Your API key
    api_key = 'your-api-key-here'
    
    # Path to image with watermark
    image_path = 'path/to/your/image.jpg'
    
    try:
        # First detect watermarks
        print(f"Detecting watermarks in {image_path}")
        detection_result = await detect_watermarks(image_path, api_key)
        
        if detection_result['watermarks']:
            print(f"Found {len(detection_result['watermarks'])} watermarks:")
            for i, watermark in enumerate(detection_result['watermarks']):
                print(f"  Watermark {i+1}:")
                print(f"    Type: {watermark['type']}")
                print(f"    Confidence: {watermark['confidence']:.2f}")
                print(f"    Location: {watermark['bbox']}")
                if 'text_content' in watermark:
                    print(f"    Text: {watermark['text_content']}")
            
            # Then remove the watermarks
            print("\nRemoving watermarks...")
            cleaned_path = await remove_watermarks(
                image_path,
                detection_result['watermarks'],
                api_key
            )
            print(f"Cleaned image saved to: {cleaned_path}")
        else:
            print("No watermarks detected")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    asyncio.run(main())
