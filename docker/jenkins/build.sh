#!/bin/bash
# Builds Docker image.

# ${@:2} skips the first argument since we don't have multiple
# Docker images, as opposed to like
# https://github.com/pietern/pytorch-dockerfiles/blob/master/build.sh
docker build \
  --no-cache \
  "${@:2}" \
  .
