import base64, io, os, tempfile, requests, shutil
from PIL import Image
from supabase import create_client, Client
import librosa, soundfile as sf

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def download_file(url: str, file_path: str) -> str:
    """Download file from URL to specified path"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {url} to {file_path}")
        return file_path
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def save_base64_image(base64_data: str, file_path: str) -> str:
    """Save base64 image to specified path"""
    try:
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        image_data = base64.b64decode(base64_data)
        Image.open(io.BytesIO(image_data)).save(file_path)
        print(f"Saved base64 image to {file_path}")
        return file_path
    except Exception as e:
        raise Exception(f"Base64 image save failed: {str(e)}")

def process_input_image(image_input: str, request_id: str) -> str:
    """Process input image and save to ComfyUI/input/{request_id}/input_image.png"""
    try:
        # Create request-specific folder
        request_folder = f"ComfyUI/input/{request_id}"
        os.makedirs(request_folder, exist_ok=True)
        target_path = f"{request_folder}/input_image.png"
        
        if image_input.startswith(('data:image', '/9j/', 'iVBORw0KGgo')):
            save_base64_image(image_input, target_path)
            return f"{request_id}/input_image.png"
        elif image_input.startswith(('http://', 'https://')):
            download_file(image_input, target_path)
            return f"{request_id}/input_image.png"
        elif os.path.exists(image_input):
            shutil.copy2(image_input, target_path)
            print(f"Copied image to {target_path}")
            return f"{request_id}/input_image.png"
        else:
            raise Exception(f"Image file not found: {image_input}")
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Image processing failed: {str(e)}")
        print(f"Full traceback:\n{error_traceback}")
        raise Exception(f"Image processing failed: {str(e)}")

def validate_audio_file(file_path: str) -> bool:
    """Validate if audio file is supported"""
    try:
        librosa.load(file_path, sr=None, duration=1)
        return True
    except Exception as e:
        print(f"Audio validation failed: {e}")
        return False

def convert_audio_to_mp3(input_path: str, output_path: str, duration_limit: int = 120) -> str:
    """Convert audio to MP3 and limit duration"""
    try:
        y, sr = librosa.load(input_path, sr=None)
        duration = len(y) / sr
        
        if duration > duration_limit:
            y = y[:int(duration_limit * sr)]
            print(f"Audio truncated from {duration:.2f}s to {duration_limit}s")
        else:
            print(f"Audio duration: {duration:.2f}s")
        
        sf.write(output_path, y, sr, format='MP3')
        print(f"Converted audio to MP3: {output_path}")
        return output_path
    except Exception as e:
        raise Exception(f"Audio conversion failed: {str(e)}")

def process_input_audio(audio_input: str, request_id: str, duration_limit: int = 120) -> str:
    """Process input audio and save to ComfyUI/input/{request_id}/input_audio.mp3"""
    try:
        # Create request-specific folder
        request_folder = f"ComfyUI/input/{request_id}"
        os.makedirs(request_folder, exist_ok=True)
        target_path = f"{request_folder}/input_audio.mp3"
        
        if audio_input.startswith(('http://', 'https://')):
            temp_path = os.path.join(tempfile.gettempdir(), f"{request_id}_temp_audio")
            download_file(audio_input, temp_path)
        elif os.path.exists(audio_input):
            temp_path = audio_input
        else:
            raise Exception(f"Audio file not found: {audio_input}")
        
        if not validate_audio_file(temp_path):
            raise Exception(f"Unsupported audio file: {temp_path}")
        
        convert_audio_to_mp3(temp_path, target_path, duration_limit)
        
        # Clean up temp file if downloaded
        if temp_path.startswith(tempfile.gettempdir()):
            try:
                os.remove(temp_path)
                print(f"Cleaned up temp audio: {temp_path}")
            except Exception as e:
                print(f"Warning: Could not clean up {temp_path}: {e}")
        
        return f"{request_id}/input_audio.mp3"
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Audio processing failed: {str(e)}")
        print(f"Full traceback:\n{error_traceback}")
        raise Exception(f"Audio processing failed: {str(e)}")

def upload_to_supabase(file_path: str, filename: str) -> str:
    """Upload file to Supabase and return public URL"""
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Determine content type
        content_type = {
            '.mp4': 'video/mp4', '.png': 'image/png', '.jpg': 'image/jpeg', 
            '.jpeg': 'image/jpeg', '.wav': 'audio/wav', '.mp3': 'audio/mpeg'
        }.get(os.path.splitext(filename)[1], 'application/octet-stream')
        
        result = supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=filename, file=file_data, file_options={"content-type": content_type}
        )
        print('result: ', result)
        
        # Handle different response formats
        if hasattr(result, 'error') and result.error:
            raise Exception(f"Supabase upload error: {result.error}")
        elif isinstance(result, dict) and result.get('error'):
            raise Exception(f"Supabase upload error: {result['error']}")
        
        # Get public URL and clean it up
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
        
        # Remove trailing question mark and any empty query parameters
        if public_url.endswith('?'):
            public_url = public_url.rstrip('?')
        
        return public_url
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Supabase upload failed: {str(e)}")
        print(f"Full traceback:\n{error_traceback}")
        raise Exception(f"Supabase upload failed: {str(e)}")

def cleanup_temp_files(file_paths: list) -> None:
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path) and file_path.startswith(tempfile.gettempdir()):
                os.remove(file_path)
                print(f"Cleaned up: {file_path}")
        except Exception as e:
            print(f"Warning: Could not clean up {file_path}: {e}")

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        return 0.0
