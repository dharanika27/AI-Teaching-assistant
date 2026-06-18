from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "tutor-rag is running"}

# def main():
#     print("Hello from tutorrag!")


# if __name__ == "__main__":
#     main()
