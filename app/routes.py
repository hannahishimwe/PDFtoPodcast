from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import StreamingResponse, PlainTextResponse


router = APIRouter(tags=["services"])

