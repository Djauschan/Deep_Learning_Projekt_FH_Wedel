# base image
FROM node:14.17.6-alpine

# set working directory
WORKDIR /app

# install app dependencies
COPY ./services/frontend/package.json ./
RUN npm install bootstrap
RUN npm install @popperjs/core
RUN npm install

# add app
COPY ./services/frontend .

# build app
RUN npm run build

# start app
CMD ["npm", "run", "serve"]