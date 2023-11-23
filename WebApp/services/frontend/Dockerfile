# base image
FROM node:18.12

# set working directory
WORKDIR /app

# install app dependencies
COPY ./services/frontend/package*.json ./
RUN npm install

# add app
COPY ./services/frontend .

# build app
RUN npm run build

# start app
CMD ["npm", "run", "serve"]