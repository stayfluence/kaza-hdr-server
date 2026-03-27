import io
import time
import numpy as np
import cv2
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

app = FastAPI(title="Kaza HDR Merge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = "https://maahssysehlwprkesokx.supabase.co"
SUPABASE_SERVICE_KEY = None  # Set via environment variable


class MergeRequest(BaseModel):
    photo_urls: list[str]
    supabase_service_key: str | None = None


MAX_DIMENSION = 1000  # Max width or height to fit in 512MB RAM with 10 images


def download_image(url: str) -> np.ndarray:
    """Download image from URL, decode, and resize to fit memory."""
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    img_array = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from {url}")

    # Resize to fit in memory - keep aspect ratio
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def mertens_hdr_merge(images: list[np.ndarray]) -> np.ndarray:
    """
    Merge multiple exposures into a single HDR image using Mertens algorithm.

    This is exposure fusion - it works directly on LDR images without
    needing camera response curves or actual HDR radiance maps.
    The algorithm weights each pixel from each exposure based on:
    - Contrast (edges/detail)
    - Saturation (colorfulness)
    - Well-exposedness (how close to middle gray)
    """
    # Resize all images to match the smallest one (in case of slight differences)
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)

    resized = []
    for img in images:
        if img.shape[0] != min_h or img.shape[1] != min_w:
            img = cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_AREA)
        resized.append(img)

    # Create Mertens merge object
    # Parameters: contrast_weight, saturation_weight, exposure_weight
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0,
    )

    # Perform the fusion
    fusion = merge_mertens.process(resized)

    # The result is in [0, 1] float range, convert to 8-bit
    # Clip values and convert
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)

    # Optional: slight contrast enhancement
    # Convert to LAB color space for better contrast adjustment
    lab = cv2.cvtColor(fusion, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Slight saturation boost
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)  # +15% saturation
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


@app.get("/health")
def health():
    return {"status": "ok", "engine": "opencv-mertens"}


@app.post("/merge")
async def merge_hdr(request: MergeRequest):
    if len(request.photo_urls) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 photos")

    if len(request.photo_urls) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 photos")

    start_time = time.time()

    # Step 1: Download all images (process max 5 for memory)
    urls = request.photo_urls
    if len(urls) > 5:
        # Pick evenly spaced images to reduce memory usage
        step = len(urls) / 5
        urls = [urls[int(i * step)] for i in range(5)]

    images = []
    for i, url in enumerate(urls):
        try:
            img = download_image(url)
            images.append(img)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download image {i + 1}: {str(e)}"
            )

    download_time = time.time() - start_time

    # Step 2: Mertens HDR merge
    merge_start = time.time()
    result = mertens_hdr_merge(images)
    merge_time = time.time() - merge_start

    # Step 3: Encode as JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    _, buffer = cv2.imencode(".jpg", result, encode_params)
    result_bytes = buffer.tobytes()

    # Step 4: Upload to Supabase Storage
    service_key = request.supabase_service_key or SUPABASE_SERVICE_KEY
    if not service_key:
        # Return the image directly if no Supabase key
        from fastapi.responses import Response
        return Response(
            content=result_bytes,
            media_type="image/jpeg",
            headers={
                "X-Download-Time": f"{download_time:.2f}s",
                "X-Merge-Time": f"{merge_time:.2f}s",
                "X-Total-Time": f"{time.time() - start_time:.2f}s",
                "X-Input-Images": str(len(images)),
            }
        )

    try:
        supabase = create_client(SUPABASE_URL, service_key)
        filename = f"hdr_{int(time.time() * 1000)}.jpg"

        supabase.storage.from_("photos").upload(
            filename,
            result_bytes,
            file_options={"content-type": "image/jpeg"},
        )

        url_data = supabase.storage.from_("photos").get_public_url(filename)

        total_time = time.time() - start_time

        return {
            "success": True,
            "url": url_data,
            "filename": filename,
            "merged_from": len(images),
            "timings": {
                "download": f"{download_time:.2f}s",
                "merge": f"{merge_time:.2f}s",
                "total": f"{total_time:.2f}s",
            },
            "resolution": f"{result.shape[1]}x{result.shape[0]}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
