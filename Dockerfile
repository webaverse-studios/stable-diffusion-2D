#syntax=docker/dockerfile:1.4
FROM zerefdragoneel/stable-diffusion-2d:v19.0.0
WORKDIR /src
COPY . /src
CMD ["python", "main.py"]