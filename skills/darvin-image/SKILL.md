---
name: darvin-image
description: Package and test a real inference service, publish its container for Darvin, and reconcile image identity or clean up task-owned Docker artifacts.
---

# Darvin inference images

Consult the separately installed Darvin platform skill and current official
artifact documentation before live image operations. Use the actual flow, not an
assumed registry push convention. Input is a verified service contract and a
baseline handoff with source, weights, dependencies and runnable tests.

## Build and test

Inspect Dockerfile, build scripts and entrypoint before execution: a build
script may also push. Check daemon access, Docker storage free space, build
context and `.dockerignore`. Reuse compatible local layers/wheels/weights.
Confirm the evaluation architecture (commonly linux/amd64), port, startup,
resource limits and any mount/network requirements against the current task.

Test real inference with actual weights and equivalent preprocessing before
containerization. Test the built image over HTTP: readiness, representative
valid inputs, output schema/labels, and invalid inputs with required errors.
Record commands, image ID and results. Mock tests do not validate deployment.
If sockets/daemon access is sandbox-blocked, obtain approval through the host.

## Publish and hand off

Reconcile existing image identities before publishing. A registry error is
not proof a tag is absent. Avoid overwriting tags; preserve intended identity
and uncertain-outcome state before writing. For interrupted publication,
inspect the exact identity and receipt before retrying. Do not invent a
digest-as-tag or assume source and published images are equivalent.

Use the documented CLI publication mode appropriate to existing image or
directory input; directory publication may build/archive/upload and require
substantial disk. Stay within authorized network and compute limits. Wait for
publication READY, save the receipt and verified repository/tag/digest mapping,
then hand these to the submission workflow. A local build or completed upload
alone does not establish platform readiness.

## Cleanup

Inspect Docker disk usage and exact containers/images first. Remove only
proven task-owned, unused objects within the user's cleanup request. Preserve
active containers, images needed by them and baseline/recovery assets unless
their removal is explicitly agreed. Avoid global prune and unrelated caches.
Record removed IDs, recovered space when measurable and recovery implications.
