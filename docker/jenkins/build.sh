#!/bin/bash
# Builds Docker image.

# Set Jenkins UID and GID if running Jenkins
if [ -n "${JENKINS:-}" ]; then
  JENKINS_UID=$(id -u jenkins)
  JENKINS_GID=$(id -g jenkins)
fi

# ${@:2} skips the first argument since we don't have multiple
# Docker images, as opposed to like
# https://github.com/pietern/pytorch-dockerfiles/blob/master/build.sh
docker build \
  --no-cache \
  --build-arg "JENKINS=${JENKINS:-}" \
  --build-arg "JENKINS_UID=${JENKINS_UID:-}" \
  --build-arg "JENKINS_GID=${JENKINS_GID:-}" \
  "${@:2}" \
  .
