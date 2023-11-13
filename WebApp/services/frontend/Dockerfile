# base image
FROM node:14.17.6-alpine

# set working directory
WORKDIR /app

# install and cache app dependencies
COPY ./services/frontend/package.json ./services/frontend/package-lock.json ./
RUN npm install

# add app
COPY ./services/frontend .

# build app
RUN npm run build

# start app
CMD ["npm", "run", "serve"]