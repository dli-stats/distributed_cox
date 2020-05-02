# Use the official image as a parent image.
FROM jupyter/datascience-notebook

ARG REPO=github.com/MrWilber/varderiv.git
ARG BRANCH=master
ARG GIT_USER
ARG GIT_PASSWORD

# Set the working directory.

# Run the command inside your image filesystem.
RUN rm -rf varderiv; mkdir varderiv
WORKDIR varderiv
RUN git init
RUN git pull https://${GIT_USER}:${GIT_PASSWORD}@${REPO} ${BRANCH}

RUN pip install -r requirements.txt

# Inform Docker that the container is listening on the specified port at runtime.
# EXPOSE 8080

# Run the specified command within the container.
# CMD [ "npm", "start" ]

# Copy the rest of your app's source code from your host to your image filesystem.
# COPY . .