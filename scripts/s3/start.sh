#!/usr/bin/env bash
# Start a local S3 service for the S3 tests, then print the environment they
# read, as KEY=VALUE lines.
#
#   set -a; eval "$(scripts/s3/start.sh localstack)"; set +a
#   cargo nextest run -p uniko-store --test integration unidb_object_store_repro::
#   cargo nextest run -p uniko-memory --test integration cloud_e2e::
#
# In CI, append the output to $GITHUB_ENV instead.
#
# Services:
#   localstack    LocalStack's S3, over plain HTTP on 127.0.0.1:4566.
#   aws-emulator  The AWS Emulator, over HTTPS on 127.0.0.1:9297 with a CA it
#                 writes at each start. AWS publishes no image, so this builds
#                 one from aws-emulator/Dockerfile. The S3 client trusts the CA
#                 through SSL_CERT_FILE, a bundle of the system roots plus it
#                 (models still download from Hugging Face over the system
#                 roots).
#
# Both honor S3's conditional writes, which uni-db's writer lease relies on.
#
# Env: CONTAINER (docker or podman; default: docker when it answers, else
# podman), BUCKET (a bucket to create; default uniko-s3-tests). The tests
# create buckets of their own.
set -euo pipefail

service="${1:-localstack}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
container="${CONTAINER:-$(command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && echo docker || echo podman)}"
bucket="${BUCKET:-uniko-s3-tests}"
name="uniko-s3-$service"
cacert=()

case "$service" in
  localstack)
    port=4566
    endpoint="http://127.0.0.1:$port"
    health="$endpoint/_localstack/health"
    "$container" rm -f "$name" >/dev/null 2>&1 || true
    "$container" run -d --name "$name" -p "127.0.0.1:$port:$port" -e SERVICES=s3 \
      docker.io/localstack/localstack:4.13.1 >&2
    ;;
  aws-emulator)
    port=9297
    endpoint="https://127.0.0.1:$port"
    health="$endpoint/__admin/health"
    home="$root/target/aws-emulator-home"
    mkdir -p "$home"
    # The image runs as uid 1000; the host user may not be (CI runners are 1001).
    chmod a+rwx "$home"
    case "$(uname -m)" in
      arm64 | aarch64) arch=aarch64 ;;
      *) arch=x86_64 ;;
    esac
    "$container" build --quiet --build-arg ARCH="$arch" -t aws-emulator "$here/aws-emulator" >&2
    "$container" rm -f "$name" >/dev/null 2>&1 || true
    "$container" run -d --name "$name" --memory=2g \
      -p "127.0.0.1:$port:$port" -e HOME="$home" -v "$home:$home" \
      aws-emulator -p "$port" -a off -t off >&2
    ca="$home/.aws/emulator/$port/ca.pem"
    for _ in $(seq 1 60); do [ -f "$ca" ] && break; sleep 1; done
    cacert=(--cacert "$ca")
    ;;
  *)
    echo "usage: $0 localstack|aws-emulator" >&2
    exit 2
    ;;
esac

for _ in $(seq 1 90); do
  if curl -sf "${cacert[@]+"${cacert[@]}"}" "$health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -sf "${cacert[@]+"${cacert[@]}"}" "$health" >/dev/null || {
  echo "$service did not become healthy" >&2
  "$container" logs "$name" >&2 || true
  exit 1
}

# The AWS Emulator serves every service on one port and tells them apart by
# the SigV4 scope, so sign. With authentication off any credentials sign.
curl -sf -o /dev/null "${cacert[@]+"${cacert[@]}"}" --aws-sigv4 "aws:amz:us-east-1:s3" \
  --user "test:test" -X PUT "$endpoint/$bucket" >&2 || true

echo "UNIKO_S3_ENDPOINT=$endpoint"
echo "UNIKO_S3_BUCKET=$bucket"
echo "AWS_ACCESS_KEY_ID=test"
echo "AWS_SECRET_ACCESS_KEY=test"
echo "AWS_REGION=us-east-1"
if [ "$service" = aws-emulator ]; then
  bundle="$home/ca-bundle.pem"
  system=""
  for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
    if [ -f "$f" ]; then system="$f"; break; fi
  done
  cat ${system:+"$system"} "$ca" >"$bundle"
  echo "UNIKO_S3_CA=$ca"
  echo "SSL_CERT_FILE=$bundle"
fi
