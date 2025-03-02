from fastapi import FastAPI
from routes import router

app = FastAPI()
app.include_router(router)

@app.get('/')
def home():
    return {"message": "License Plate Recognition API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)