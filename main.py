from fastapi import FastAPI, File, UploadFile, Response
import uvicorn
import os
from fastapi.responses import FileResponse
from model import *
from starlette.middleware.cors import CORSMiddleware


names = {
    0: 'CBM.37.060',
    1: 'CBM.37.060A',
    2: 'СВП-120.00.060',
    3: 'СВП120.42.020',
    4: 'СВП120.42.030',
    5: 'CK20.01.01.01.406',
    6: 'CK20.01.01.02.402',
    7: 'СПО250.14.190',
    8: 'CK30.01.01.02.402',
    9: 'CK30.01.01.03.403',
    10: 'CK50.01.01.404',
    11: 'CK50.02.01.411',
    12: 'ЗВТ86.103К-02',
    13: 'CS120.01.413',
    14: 'CS120.07.442',
    15: 'CS150.01.427-01',
    16: 'SU160.00.404',
    17: 'SU80.01.426',
    18: 'SU80.10.409A',
    19: 'ЗВТ86.103К-02',
    20: 'СВМ.37.060',
    21: 'СВМ.37.060А',
    22: 'СВП-120.00.060',
    23: 'СВП120.42.020',
    24: 'СВП120.42.030',
    25: 'СК20.01.01.01.406',
    26: 'СК20.01.01.02.402',
    27: 'СК30.01.01.02.402',
    28: 'СК30.01.01.03.403',
    29: 'СК50.01.01.404',
    30: 'СК50.02.01.411',
    31: 'СПО250.14.190'}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/image/")
async def upload_image(image: UploadFile = File(...)):
    contents = await image.read()
    with open(f"predict.jpg", 'wb') as file:
        file.write(contents)
    try:
        model_small.predict('predict.jpg', save=True, conf=0.5)
        return {"message": "success"}
    except:
        return {"message": "error"}


@app.put("/image/")
async def upload_image(model: str, image: UploadFile = File(...)):
    contents = await image.read()
    with open(f"predict.jpg", 'wb') as file:
        file.write(contents)
    try:
        if model == 'small':
            results = model_small('predict.jpg')
        if model == 'nano':
            results = model_nano('predict.jpg')
        # else:
        #     results = model_small('predict.jpg')
        ans = []
        for result in results:
            x, y, w, h = result.boxes.xywhn[0].tolist()
            conf = float(result.boxes.conf[0])
            class_id = int(result.boxes.cls[0])
            name = names[class_id]
            ans.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                "confidence": conf,
                'class': name

            })
        return {"message": ans}
    except:
        return {"message": "error"}



@app.get("/getimage/")
async def im_get():
    folder_path = "runs/detect"
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    max_num = 0
    for fold in folders:
        if fold != 'predict':
            num = int(fold.strip('predict'))
            max_num = max(num, max_num)
    filename = f"runs/detect/predict{str(max_num)}/predict.jpg"
    # filename = f"runs/detect/predict/predict.jpg"
    return FileResponse(filename)


if __name__ == '__main__':
    uvicorn.run(app, port=8000)
