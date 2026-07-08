# MinkaFX

The Guido-style half of the Minka hybrid shell: per-frame-hot, geometrically
simple, **inputless** overlays rendered with wgpu on wlr-layer-shell overlay
surfaces, one per output. MinkaShell (Quickshell) owns everything with text,
input, or services; MinkaFX owns what animates every frame. Both are peers on
ShojiWM's NDJSON IPC socket via the MinkaIPC crate.

## Status — milestone M3

- [x] MinkaIPC: socket thread + calloop channel, auto-reconnect (1s), never
      blocks the render loop (rule R1).
- [x] Snap-zone preview: consumes `snap.preview {monitor, rect|null, kind}`,
      springs the rounded red rect toward the target client-side (rule R3 —
      transmit state, not frames), fades in/out, sleeps when settled (zero
      frames rendered while idle).
- [ ] OSDs (volume/brightness pop-overlays) — next.
- ~~Dock reveal effect~~ — dropped: the dock is a persistent taskbar now
  (Sophie, 8/7/2026), so there is no reveal to animate.

## Behavior notes

- Surfaces have an **empty input region** and never take keyboard focus, so a
  crash or hang here can never wedge the session.
- Frame callbacks drive the animation; when springs settle the process stops
  rendering entirely until the next broadcast.
- Snap rect is treated as output-local logical coordinates. If testing shows
  the preview offset from the drop zone, the compositor is sending
  workspace-area coords and an offset needs adding here.
- The ShojiWM config spawns `target/release/MinkaFX` guarded by `[ -x ... ]`,
  logging to `/tmp/minkafx.log`; if the binary is missing the spawn is a
  silent no-op.

## Building

```sh
cargo build --release   # binary: target/release/MinkaFX
```
