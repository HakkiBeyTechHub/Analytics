from predict_class import Predictor, Utils
from fastapi import FastAPI, File

predictor = Predictor()
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: bytes = File(...)):
    imagem = Utils.read_image(file)
    prediction = predictor.predict(imagem)
    return prediction