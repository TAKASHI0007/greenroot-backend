# backend/main.py
# Phase 14 + GEE統合版
# Railway デプロイ対応

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import jwt
import os
import uuid
import hashlib
import logging
import json

# ========== ロギング設定 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ========== 環境変数 ==========
SECRET_KEY          = os.getenv("SECRET_KEY", "change-this-in-production")
ALGORITHM           = "HS256"
TOKEN_EXPIRE_H      = int(os.getenv("TOKEN_EXPIRE_HOURS", "24"))
ALLOWED_ORIGINS     = os.getenv("ALLOWED_ORIGINS", "*").split(",")
PORT                = int(os.getenv("PORT", "8000"))
GEE_SERVICE_ACCOUNT = os.getenv("GEE_SERVICE_ACCOUNT", "greenroot-gee@warm-cycle-503404-b1.iam.gserviceaccount.com")
GEE_KEY_JSON        = os.getenv("GEE_KEY_JSON", "")

# ========== GEE初期化 ==========
GEE_ENABLED = False

# After
def init_gee():
    global GEE_ENABLED
    if not GEE_SERVICE_ACCOUNT or not GEE_KEY_JSON:
        logger.warning("⚠️ GEE credentials not set → using mock data")
        return False
    try:
        import ee

        # 文字列またはdictの両方に対応
        if isinstance(GEE_KEY_JSON, str):
            key_data = json.loads(GEE_KEY_JSON)
        else:
            key_data = GEE_KEY_JSON

        credentials = ee.ServiceAccountCredentials(
            GEE_SERVICE_ACCOUNT, key_data=key_data
        )
        ee.Initialize(credentials)
        GEE_ENABLED = True
        logger.info("✅ Google Earth Engine initialized")
        return True
    except Exception as e:
        logger.error(f"❌ GEE init failed: {e}")
        return False

# ========== アプリ初期化 ==========
app = FastAPI(
    title="GreenRoot AgriTech API",
    description="AgriSense + CarbonLedger + GEE Satellite",
    version="2.1.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

security = HTTPBearer(auto_error=False)

@app.on_event("startup")
async def startup_event():
    init_gee()

# ========== インメモリDB ==========
_users:       Dict[str, dict] = {}
_detections:  List[dict]      = []
_diagnoses:   List[dict]      = []
_actions:     List[dict]      = []
_mrv_records: List[dict]      = []

# ========== Auth Utils ==========
def hash_password(password: str) -> str:
    return hashlib.sha256(
        (password + SECRET_KEY).encode()
    ).hexdigest()

def create_token(data: dict) -> str:
    payload = {
        **data,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_H),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Token required")
    try:
        return jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ========== Pydantic Models ==========
class UserRegister(BaseModel):
    email:     str
    password:  str
    farmer_id: str
    farm_name: str
    location:  Optional[str]   = "Armenia"
    area_ha:   Optional[float] = 1.0

class UserLogin(BaseModel):
    email:    str
    password: str

class DetectionUpload(BaseModel):
    farmer_id:  str
    detections: List[dict] = []
    latitude:   float = 0.0
    longitude:  float = 0.0
    timestamp:  Optional[str] = None

class DiagnosisSave(BaseModel):
    farmer_id:       str
    disease_name:    str
    confidence:      float
    severity:        str
    soil_ph:         Optional[float] = 7.0
    soil_moisture:   Optional[float] = 0.5
    recommendations: List[str]       = []
    latitude:        Optional[float] = 0.0
    longitude:       Optional[float] = 0.0

class BlockchainAction(BaseModel):
    farmer_id:   str
    action_id:   str
    action_type: str
    latitude:    float
    longitude:   float
    description: str
    tx_hash:     Optional[str] = None

class MRVSubmission(BaseModel):
    project_id:         str
    farmer_id:          str
    area_ha:            float
    ndvi_mean:          float
    ndmi_mean:          float
    carbon_sequestered: float
    measurement_date:   str
    vcu_quantity:       int
    blockchain_hash:    Optional[str] = None

class SatelliteDataSync(BaseModel):
    farmer_id:   str
    latitude:    float
    longitude:   float
    ndvi:        float
    ndmi:        float
    cloud_cover: float
    image_date:  str

class BatchSync(BaseModel):
    detections: List[dict] = []
    diagnoses:  List[dict] = []
    actions:    List[dict] = []

# ========== GEE Sentinel-2 計算 ==========
ARMENIA_REGIONS = {
    'ararat':      (39.8136, 44.5152),
    'vayots_dzor': (39.7270, 45.1833),
    'areni':       (39.7108, 45.1869),
    'kotayk':      (40.2200, 44.5600),
}

def _ndvi_status(ndvi: float) -> str:
    if ndvi >= 0.6: return 'Excellent'
    if ndvi >= 0.4: return 'Good'
    if ndvi >= 0.2: return 'Moderate'
    return 'Poor'

def _mock_satellite(lat: float, lon: float, days: int = 30) -> dict:
    """GEE未接続時のアルメニア季節推定値"""
    month = datetime.utcnow().month
    if   5 <= month <= 7: ndvi = 0.55 + (month - 5) * 0.05
    elif 8 <= month <= 9: ndvi = 0.50 + (9 - month) * 0.05
    elif 3 <= month <= 4: ndvi = 0.25 + (month - 3) * 0.10
    else:                 ndvi = 0.15
    return {
        'source':      'Mock (GEE not configured)',
        'real_data':   False,
        'latitude':    lat,
        'longitude':   lon,
        'ndvi':        round(ndvi, 4),
        'ndwi':        round(ndvi * 0.6, 4),
        'evi':         round(ndvi * 0.85, 4),
        'ndre':        round(ndvi * 0.9, 4),
        'bsi':         round(0.1 - ndvi * 0.1, 4),
        'ndvi_status': _ndvi_status(ndvi),
        'cloud_cover': 12.0,
        'image_date':  datetime.utcnow().strftime('%Y-%m-%d'),
        'image_count': 0,
        'period_days': days,
        'timestamp':   datetime.utcnow().isoformat(),
    }

def calculate_ndvi_gee(
    latitude:  float,
    longitude: float,
    days_back: int = 30,
) -> dict:
    """GEE で Sentinel-2 NDVI/NDWI/EVI/NDRE/BSI を計算"""
    if not GEE_ENABLED:
        return _mock_satellite(latitude, longitude, days_back)
    try:
        import ee
        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        point = ee.Geometry.Point([longitude, latitude])
        aoi   = point.buffer(5000)  # 5km バッファ

        def mask_clouds(image):
            qa = image.select('QA60')
            mask = (
                qa.bitwiseAnd(1 << 10).eq(0)
                .And(qa.bitwiseAnd(1 << 11).eq(0))
            )
            return image.updateMask(mask).divide(10000)

        col = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
            )
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .map(mask_clouds)
        )
        count = col.size().getInfo()
        logger.info(f"📡 GEE: {count} images ({latitude:.4f}, {longitude:.4f})")
        if count == 0:
            return _mock_satellite(latitude, longitude, days_back)

        composite = col.median()
        ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI')
        ndre = composite.normalizedDifference(['B8A', 'B5']).rename('NDRE')
        evi  = composite.expression(
            '2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',
            {'NIR': composite.select('B8'),
             'RED': composite.select('B4'),
             'BLUE': composite.select('B2')},
        ).rename('EVI')
        bsi = composite.expression(
            '((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE))',
            {'SWIR': composite.select('B11'),
             'RED':  composite.select('B4'),
             'NIR':  composite.select('B8'),
             'BLUE': composite.select('B2')},
        ).rename('BSI')

        combined = ndvi.addBands(ndwi).addBands(ndre).addBands(evi).addBands(bsi)
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
        ).getInfo()

        latest       = col.sort('system:time_start', False).first()
        latest_date  = latest.date().format('YYYY-MM-dd').getInfo()
        latest_cloud = latest.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        ndvi_val     = stats.get('NDVI', 0.0) or 0.0

        return {
            'source':      'Google Earth Engine / Sentinel-2',
            'real_data':   True,
            'latitude':    latitude,
            'longitude':   longitude,
            'ndvi':        round(ndvi_val, 4),
            'ndwi':        round(stats.get('NDWI', 0.0) or 0.0, 4),
            'evi':         round(stats.get('EVI',  0.0) or 0.0, 4),
            'ndre':        round(stats.get('NDRE', 0.0) or 0.0, 4),
            'bsi':         round(stats.get('BSI',  0.0) or 0.0, 4),
            'ndvi_status': _ndvi_status(ndvi_val),
            'cloud_cover': round(latest_cloud or 0.0, 1),
            'image_date':  latest_date,
            'image_count': count,
            'period_days': days_back,
            'timestamp':   datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ GEE error: {e}")
        return _mock_satellite(latitude, longitude, days_back)

# ========== Health ==========
@app.get("/")
async def root():
    return {
        "message": "GreenRoot API v2.1.0",
        "status":  "running",
        "gee":     GEE_ENABLED,
    }

@app.get("/api/health")
async def health():
    return {
        "status":      "healthy",
        "version":     "2.1.0",
        "gee_enabled": GEE_ENABLED,
        "timestamp":   datetime.utcnow().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "production"),
    }

@app.get("/api/version")
async def version():
    return {"version": "2.1.0", "phase": 14, "gee": GEE_ENABLED}

# ========== Auth ==========
@app.post("/api/auth/register")
async def register(user: UserRegister):
    if user.email in _users:
        raise HTTPException(status_code=400, detail="Email already registered")
    _users[user.email] = {
        "id":            str(uuid.uuid4()),
        "email":         user.email,
        "password_hash": hash_password(user.password),
        "farmer_id":     user.farmer_id,
        "farm_name":     user.farm_name,
        "location":      user.location,
        "area_ha":       user.area_ha,
        "created_at":    datetime.utcnow().isoformat(),
    }
    logger.info(f"New user: {user.farmer_id}")
    token = create_token({"sub": user.email, "farmer_id": user.farmer_id})
    return {"success": True, "token": token,
            "farmer_id": user.farmer_id, "farm_name": user.farm_name}

@app.post("/api/auth/login")
async def login(user: UserLogin):
    stored = _users.get(user.email)
    if not stored or stored["password_hash"] != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token({"sub": user.email, "farmer_id": stored["farmer_id"]})
    return {"success": True, "token": token,
            "farmer_id": stored["farmer_id"], "farm_name": stored["farm_name"]}

@app.post("/api/auth/refresh")
async def refresh(payload: dict = Depends(verify_token)):
    token = create_token({
        "sub": payload["sub"],
        "farmer_id": payload.get("farmer_id", ""),
    })
    return {"token": token}

# ========== Detections ==========
@app.post("/api/detections/upload")
async def upload_detection(
    data: DetectionUpload, _: dict = Depends(verify_token)
):
    det_id = str(uuid.uuid4())
    _detections.append({
        "id": det_id, **data.dict(),
        "timestamp":  data.timestamp or datetime.utcnow().isoformat(),
        "synced_at":  datetime.utcnow().isoformat(),
    })
    return {"success": True, "detection_id": det_id}

@app.get("/api/detections/{farmer_id}")
async def get_detections(farmer_id: str, _: dict = Depends(verify_token)):
    results = [d for d in _detections if d["farmer_id"] == farmer_id]
    return {"detections": results, "count": len(results)}

# ========== Diagnoses ==========
@app.post("/api/diagnoses/save")
async def save_diagnosis(
    data: DiagnosisSave, _: dict = Depends(verify_token)
):
    diag_id = str(uuid.uuid4())
    _diagnoses.append({
        "id": diag_id, **data.dict(),
        "created_at": datetime.utcnow().isoformat(),
    })
    return {"success": True, "diagnosis_id": diag_id}

@app.get("/api/diagnoses/{farmer_id}")
async def get_diagnoses(farmer_id: str, _: dict = Depends(verify_token)):
    results = [d for d in _diagnoses if d["farmer_id"] == farmer_id]
    return {"diagnoses": results, "count": len(results)}

# ========== Blockchain ==========
@app.post("/api/blockchain/actions")
async def record_action(
    data: BlockchainAction, _: dict = Depends(verify_token)
):
    _actions.append({
        "id": str(uuid.uuid4()), **data.dict(),
        "created_at": datetime.utcnow().isoformat(),
    })
    return {"success": True, "action_id": data.action_id}

@app.get("/api/blockchain/actions/{farmer_id}")
async def get_actions(farmer_id: str, _: dict = Depends(verify_token)):
    results = [a for a in _actions if a["farmer_id"] == farmer_id]
    return {"actions": results, "count": len(results)}

# ========== Satellite（GEE統合） ==========
@app.post("/api/satellite/sync")
async def sync_satellite(
    data: SatelliteDataSync, _: dict = Depends(verify_token)
):
    return {"success": True, "record_id": str(uuid.uuid4())}

@app.get("/api/satellite/latest/{farmer_id}")
async def get_satellite(
    farmer_id: str,
    lat:  float = 39.8136,
    lon:  float = 44.5152,
    days: int   = 30,
    _: dict = Depends(verify_token),
):
    """GEE実データ取得（フォールバック: モック）"""
    data = calculate_ndvi_gee(lat, lon, days)
    data['farmer_id'] = farmer_id
    return data

@app.get("/api/satellite/region/{region}")
async def get_satellite_by_region(
    region: str,
    days:   int  = 30,
    _: dict = Depends(verify_token),
):
    """アルメニア地域別Sentinel-2データ取得"""
    if region not in ARMENIA_REGIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown region '{region}'. "
                   f"Valid: {list(ARMENIA_REGIONS.keys())}",
        )
    lat, lon = ARMENIA_REGIONS[region]
    data = calculate_ndvi_gee(lat, lon, days)
    data['region'] = region
    return data

@app.get("/api/satellite/regions")
async def list_regions(_: dict = Depends(verify_token)):
    """利用可能な地域一覧"""
    return {
        "regions": [
            {"id": "ararat",      "name": "Ararat Valley",
             "lat": 39.8136, "lon": 44.5152},
            {"id": "vayots_dzor", "name": "Vayots Dzor",
             "lat": 39.7270, "lon": 45.1833},
            {"id": "areni",       "name": "Areni Village",
             "lat": 39.7108, "lon": 45.1869},
            {"id": "kotayk",      "name": "Kotayk Region",
             "lat": 40.2200, "lon": 44.5600},
        ]
    }

# ========== Verra MRV ==========
@app.post("/api/verra/submit-mrv")
async def submit_mrv(
    data: MRVSubmission, _: dict = Depends(verify_token)
):
    mrv_id = str(uuid.uuid4())
    _mrv_records.append({
        "id": mrv_id, **data.dict(),
        "status":       "submitted",
        "submitted_at": datetime.utcnow().isoformat(),
    })
    logger.info(f"MRV: {data.project_id}, {data.vcu_quantity} VCUs")
    return {
        "success": True, "mrv_id": mrv_id,
        "status": "submitted", "vcu_quantity": data.vcu_quantity,
    }

# ========== Analytics ==========
@app.get("/api/analytics/summary/{farmer_id}")
async def get_analytics(farmer_id: str, _: dict = Depends(verify_token)):
    diag = [d for d in _diagnoses  if d["farmer_id"] == farmer_id]
    det  = [d for d in _detections if d["farmer_id"] == farmer_id]
    act  = [a for a in _actions    if a["farmer_id"] == farmer_id]
    mrv  = [r for r in _mrv_records if r["farmer_id"] == farmer_id]
    total_vcu    = sum(r.get("vcu_quantity", 0)          for r in mrv)
    total_carbon = sum(r.get("carbon_sequestered", 0.0)  for r in mrv)
    return {
        "farmer_id": farmer_id,
        "summary": {
            "total_detections":          len(det),
            "total_diagnoses":           len(diag),
            "total_blockchain_actions":  len(act),
            "total_vcu_issued":          total_vcu,
            "total_carbon_sequestered_t": round(total_carbon, 2),
            "estimated_revenue_usd":     round(total_vcu * 12.5, 2),
        },
        "last_updated": datetime.utcnow().isoformat(),
    }

# ========== Batch Sync ==========
@app.post("/api/sync/batch")
async def batch_sync(data: BatchSync, _: dict = Depends(verify_token)):
    synced = {"detections": 0, "diagnoses": 0, "actions": 0}
    for d in data.detections:
        _detections.append({**d, "synced_at": datetime.utcnow().isoformat()})
        synced["detections"] += 1
    for d in data.diagnoses:
        _diagnoses.append({**d, "synced_at": datetime.utcnow().isoformat()})
        synced["diagnoses"] += 1
    for a in data.actions:
        _actions.append({**a, "synced_at": datetime.utcnow().isoformat()})
        synced["actions"] += 1
    logger.info(f"Batch sync: {synced}")
    return {
        "success": True, "synced": synced,
        "timestamp": datetime.utcnow().isoformat(),
    }

# ========== エントリポイント ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
