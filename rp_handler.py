import base64
import io
import random
import uuid
from workflow import DetailerWorkflow
from rp_utils import process_input_image, cleanup_temp_files
import runpod

# Initialize the workflow once at startup
workflow = DetailerWorkflow()


def handler(event):
    try:
        # Extract input parameters
        input_data = event.get("input", {})
        
        # Generate or use provided request_id
        request_id = input_data.get("request_id", str(uuid.uuid4()))
        
        # Validate required parameters
        if "image" not in input_data:
            return {
                "request_id": request_id,
                "error": "Missing required parameter: image"
            }
        
        # Extract parameters with defaults
        user_prompt = input_data.get("user_prompt", "")
        image_input = input_data["image"]
        fixed_positive = input_data.get("fixed_positive", 
            "a face with realistic skin texture, blemishes\n\n\nraw photo, (realistic:1.5), (skin details, skin texture:0.45), (skin pores:0.45), (skin imperfections:0.19), (eyes details:0.44), selfie-style amateur photography, film grain")
        
        user_negative = input_data.get("user_negative", "")
        
        fixed_negative = input_data.get("fixed_negative", 
            "blurry, out of focus, cartoon, poor quality,  bokkeh, depth of field, tan, sunburned, professional photo, studio lighting, NSFW, nudity")
        
        # Use provided seed or generate random one
        seed = input_data.get("seed")
        if seed is None:
            seed = random.randint(1, 2**64)
        
        denoise = input_data.get("denoise", 0.21)
        
        # Process image input (base64 or URL) using rp_utils
        try:
            image_path = process_input_image(image_input, request_id)
            print('Saved image to', image_path)
        except Exception as e:
            return {
                "request_id": request_id,
                "error": f"Failed to process image: {str(e)}"
            }
        
        # Validate that all text prompts are strings
        text_params = {
            "user_prompt": user_prompt,
            "fixed_positive": fixed_positive,
            "user_negative": user_negative,
            "fixed_negative": fixed_negative
        }
        
        for param_name, param_value in text_params.items():
            if not isinstance(param_value, str):
                return {
                    "request_id": request_id,
                    "error": f"{param_name} must be a string"
                }
        
        
        # Validate numeric parameters
        if not isinstance(denoise, (int, float)) or not (0 <= denoise <= 1):
            return {
                "request_id": request_id,
                "error": "denoise must be a number between 0 and 1"
            }
        
        if not isinstance(seed, int) or seed < 0:
            return {
                "request_id": request_id,
                "error": "seed must be a positive integer"
            }
        
        # Log request details before starting generation
        payload = {
            "request_id": request_id,
            "user_prompt": user_prompt,
            "fixed_positive": fixed_positive,
            "user_negative": user_negative,
            "fixed_negative": fixed_negative,
            "seed": seed,
            "denoise": denoise,
            "image_path": image_path
        }
        print(f"Starting detailer generation for request_id: {request_id}")
        print(f"Payload: {payload}")
        
        # Run the workflow
        result_image = workflow(
            user_prompt=user_prompt,
            fixed_positive=fixed_positive,
            user_negative=user_negative,
            fixed_negative=fixed_negative,
            seed=seed,
            denoise=denoise,
            image_path=image_path
        )
        
        # Convert PIL Image to base64
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Log image size
        img_size_mb = len(img_base64) / (1024 * 1024)
        print(f"Base64 image size: {img_size_mb:.2f} MB")
        
        # Clean up temporary files using rp_utils
        cleanup_temp_files([image_path])
        
        # Log completion before sending result
        print(f"Detailer generation completed for request_id: {request_id}")
        
        return {
            "request_id": request_id,
            "success": True,
            "image": img_base64
        }
        
    except Exception as e:
        # Clean up temporary files using rp_utils
        if 'image_path' in locals() and image_path:
            cleanup_temp_files([image_path])
        
        return {
            "request_id": request_id if 'request_id' in locals() else str(uuid.uuid4()),
            "error": f"Processing failed: {str(e)}"
        }

# Example usage for testing
if __name__ == "__main__":
    print("Running serverless detailer generation service")
    runpod.serverless.start({"handler": handler})
