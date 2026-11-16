# Diorama Velocity

**Diorama Velocity** is a performance-focused fork of [Diorama](https://github.com) engineered to eliminate *render-thread* frame spikes, synchronous GPU stalls, and chunk-loading stutters during active exploration and fast camera movement in modern Minecraft (NeoForge / Fabric).

> **Note:** While the upstream mod achieves high static framerates, rapid chunk loading, cascading light engine updates, and synchronous OpenGL calls can cause noticeable micro-stutters (*frametime spikes*) while moving or turning the camera. Diorama Velocity resolves these bottlenecks without sacrificing visual fidelity.

---

## Key Optimizations

### 1. Zero-Allocation GPU Readbacks & Buffer Orphaning (`GlBakeBackend`)
* **Reusable PBO Pool:** Replaces per-tile `glGenBuffers` / `glDeleteBuffers` calls with a persistent Pixel Buffer Object (PBO) pool, eliminating driver memory allocation stalls during gameplay.
* **Smart VBO/UBO Capacity Reuse:** Uses explicit buffer orphaning (`glBufferData` + `glBufferSubData`) only when capacity grows, preventing implicit OpenGL pipeline synchronization stalls.
* **Throttled Readback Polling:** Caps completed tile readbacks to a maximum of 2 tiles per frame in `poll()` to keep CPU mapping overhead predictable.

### 2. Strict Frame-Time Budgeting (`GpuTileBaker`)
* **Single-Tile Frame Cap:** Enforces a strict limit of **1 GPU tile bake per frame** during active gameplay, preventing accumulated frame budgets from triggering multi-tile burst stalls when entering new chunks.
* **Tighter Frame Share:** Reduces active render-thread bake budget from 10% to 4% (and background fill to 1.5%) to prioritize smooth camera rotation and world rendering.

### 3. Proximity-Based Adaptive Chunk Snapshots (`MapSession`)
* **Adaptive Snapshot Throttling:** Lowers the maximum chunk snapshots per tick from `24` down to `6` (and dynamically scales down to `2` under heavy queue congestion).
* **Closest-First Prioritization:** Processes pending chunk snapshots sorted by proximity to the player rather than strict FIFO order, ensuring immediate surroundings update first without choking the client tick.
* **Event Debouncing:** Increases batching windows (`500 ms` for block/light updates, `900 ms` for newly loaded chunks) and reduces periodic chunk sweep frequency (every `80` ticks) to coalesce cascading light engine updates (e.g., alongside *ScalableLux* or *Starlight*).

### 4. Non-Blocking Fog Mask & Layer Caching (`MapViewport`)
* **Fully Asynchronous Mask Finishing:** Removes blocking `CompletableFuture.join()` calls from the render thread during player movement, offloading mask blurring and soft-edge calculations cleanly to `MASK_WORKER`.
* **Throttled Fog Layer Drift:** Updates the half-resolution `FogLayer` drift every 4 game ticks instead of every single tick, cutting fragment shader overhead during rapid camera rotation.

---

## Building from Source

This repository uses [Stonecutter](https://kikugie.dev) to target multiple Minecraft versions from a single codebase.

1. Clone the repository and open it in **IntelliJ IDEA** (with **JDK 21+**).
2. To build a specific target version (for example, Minecraft `26.1.2` on NeoForge), run:
   ```bash
   ./gradlew :26.1.2-neoforge:build
   ```
3. Locate the compiled mod `.jar` inside:
   ```text
   versions/26.1.2-neoforge/build/libs/
   ```
   *(Use the standard `.jar` file, excluding `-dev` or `-sources` artifacts).*

---

## Credits & License

* **Original Mod:** Created by [SunkenInTime](https://github.com).
* **License:** Follows the original project's license terms (see `LICENSE`).
