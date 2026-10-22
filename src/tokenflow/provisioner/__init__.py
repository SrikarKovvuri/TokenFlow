"""Worker and GPU provisioning.

This package will eventually allocate GPUs and start inference processes.
Phase 2 does not do that. Start vLLM separately, then register its URL with
the worker manager.
"""
