# base image
FROM node:18.12

# set working directory
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/npm \
    --mount=type=bind,source=package.json,target=package.json \
    npm install

COPY . .

CMD ["npm", "run", "build"]
