from fastapi import FastAPI, File, UploadFile
from AImodel.model.gardener_model import VirtualGardener
from pydantic import BaseModel

app = FastAPI()

PREDICTIONS_PATH = "predictions"
model = VirtualGardener(path_to_save=PREDICTIONS_PATH)

class PredictionResponse(BaseModel):
    class_name: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(image: UploadFile = File(...)):
    """
    Принимает изображение и возвращает предсказание модели.
    """
    image_data = await image.read() # Читаем содержимое файла в память

    # Получаем предсказание от модели напрямую
    class_name = model.predict(image_data=image_data)

    return {"class_name": class_name}