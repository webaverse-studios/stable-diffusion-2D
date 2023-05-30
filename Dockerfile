FROM zerefdragoneel/stable-diffusion-2d:v7.0.0
WORKDIR /src
RUN rm -rf ./*
ADD . .