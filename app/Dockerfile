# Lekki obraz bazowy z Pythonem - "slim" = bez zbędnych narzędzi, mniejszy rozmiar
FROM python:3.11-slim

# Folder roboczy WEWNĄTRZ kontenera - wszystko poniżej dzieje się w /app
WORKDIR /app

# Najpierw kopiujemy TYLKO requirements.txt i instalujemy zależności.
# Dzięki temu, jeśli zmienisz tylko predict.py, Docker nie musi przeinstalowywać
# wszystkich pakietów od nowa (wykorzystuje cache warstwy z pip install).
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Teraz kopiujemy kod i zapisane modele
COPY app/predict.py .
COPY models ./models

# Komenda uruchamiana, gdy ktoś odpali kontener
CMD ["python", "predict.py"]
