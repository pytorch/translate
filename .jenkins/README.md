# Jenkins continuous integration test

## Usage

Jenkins will automatically trigger a new build/test under
https://ci.pytorch.org/jenkins/job/translate-builds/job/translate-xenial-cuda9-cudnn7-py3-build-test/
whenever a pull request is opened or updated, and display the results on the
pull request. If you need to re-run the test due to infrastructure issues
or non-code-related changes, you can manually trigger a re-test by commenting
on the pull request "@pytorchbot retest this please" (this should usually not
be necessary).

## Implementation details

After Jenkins has patched the appropriate version of the PyTorch Translate
repo on a Docker image, it calls `build.sh` via
https://github.com/pytorch/ossci-job-dsl/blob/master/src/jobs/translate.groovy
to build/install PyTorch Translate and run tests. `build.sh` should be
self-sufficient given the repo code, and not require any internet access
to build and run the tests. (Note that fairseq's
[setup.py](https://github.com/pytorch/fairseq/blob/main/setup.py#L42)
may try to fetch some
[required packages](https://github.com/pytorch/fairseq/blob/main/requirements.txt).
They're currently already installed in our Docker image
[requirements](https://github.com/pytorch/translate/blob/master/docker/jenkins/install_prereqs.sh) -
though that may need to be updated in the future.)

If you need to download other packages or dependencies, consider adding them
to the Docker image instead, in the `../docker/jenkins/Dockerfile` used
by `../docker/jenkins/build.sh`.
