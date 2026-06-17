"""Search-kernel primitives for diversity-preserving Nexus evolution.

Submodules are intentionally not eagerly imported here: ``semantic_dedupe`` uses
``search_kernel.fingerprints`` and harvesting uses ``semantic_dedupe``, so eager
package-level imports would create an import cycle.
"""

__all__: list[str] = []
