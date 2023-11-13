# base image
FROM node:14.17.6-alpine

# set working directory
WORKDIR /app

# install app dependencies
COPY ./services/frontend/package.json ./
RUN npm cache clean -force
RUN npm install npm@latest -g

# add app
COPY ./services/frontend .

# build app
RUN npm run build

# start app
CMD ["npm", "run", "serve"]