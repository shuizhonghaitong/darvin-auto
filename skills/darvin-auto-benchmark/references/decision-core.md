# Offline decision core

The experimental `lib/` modules plan and validate decisions offline; they do
not execute live platform operations. Inspect implementation and tests before
integrating them into another executor.

- Registry lookup failure or a missing CLI does not prove an image is absent.
- Deduplication requires strong identity; reconcile ambiguous writes before retrying.
- Matching contract anchors alone does not establish a complete contract.
- Task status and evaluation job status are distinct; query scores separately.
- State uses locking and atomic writes. The pipeline defaults to no writes and
  has no live executor.
