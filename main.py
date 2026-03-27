import io
import time
import gc
import numpy as np
import cv2
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Kaza HDR Merge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MergeRequest(BaseModel):
    photo_urls: list[str]


# 2000px max - good quality while staying under 512MB RAM
# 5 images at 2000x1500x3 = ~45MB raw, Mertens needs ~4x = ~180MB, safe margin
MAX_DIMENSION = 2000


def download_image(url: str) -> np.ndarray:
    """Download image from URL, decode, and resize to fit memory."""
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    img_array = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from {url}")

    # Free the raw download buffer
    del img_array, response
    gc.collect()

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
    Merge multiple exposures using Mertens exposure fusion.
    Clean output, no aggressive post-processing.
    """
    # Align all images to same size
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)

    resized = []
    for img in images:
        if img.shape[0] != min_h or img.shape[1] != min_w:
            img = cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_AREA)
        resized.append(img)

    # Align images to compensate hand movement between shots
    align = cv2.createAlignMTB()
    align.process(resized, resized)

    # Mertens exposure fusion
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0,
    )

    fusion = merge_mertens.process(resized)

    # Free input images
    del resized
    gc.collect()

    # Convert from [0,1] float to 8-bit
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)

    # Light contrast boost only via gentle CLAHE (not aggressive)
    lab = cv2.cvtColor(fusion, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return result


@app.get("/health")
def health():
    return {"status": "ok", "engine": "opencv-mertens", "max_px": MAX_DIMENSION}


@app.post("/merge")
async def merge_hdr(request: MergeRequest):
    if len(request.photo_urls) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 photos")

    if len(request.photo_urls) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 photos")

    start_time = time.time()

    # Pick max 5 evenly spaced images to fit in memory
    urls = request.photo_urls
    if len(urls) > 5:
        step = len(urls) / 5
        urls = [urls[int(i * step)] for i in range(5)]

    # Download all images
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

    # Mertens HDR merge
    merge_start = time.time()
    result = mertens_hdr_merge(images)

    # Free input images
    del images
    gc.collect()

    merge_time = time.time() - merge_start

    # Encode as high quality JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    _, buffer = cv2.imencode(".jpg", result, encode_params)
    result_bytes = buffer.tobytes()

    del result, buffer
    gc.collect()

    total_time = time.time() - start_time

    return Response(
        content=result_bytes,
        media_type="image/jpeg",
        headers={
            "X-Download-Time": f"{download_time:.2f}s",
            "X-Merge-Time": f"{merge_time:.2f}s",
            "X-Total-Time": f"{total_time:.2f}s",
            "X-Input-Images": str(len(urls)),
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
