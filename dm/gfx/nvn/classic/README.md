# Classic Metal NVN execution layer

`nvn_mtl.m` here is the classic-Metal (macOS 15 / iOS 18 floor) execution layer of the NVN
null device, built only with `DM_NVN_M3=1` (see `dm/gfx/README.md`). It implements the same
replay contract as the Metal 4 layer (`../nvn_mtl.m`) on the classic Metal API: ordinary
command queue / command buffers, render/compute/blit encoders, per-slot buffer bindings, and
automatic hazard tracking.

## Synchronization

All NVN submissions are serialized onto one Metal queue. A command-buffer wait for a
previously submitted fence, or for a command-buffer fence recorded earlier in the same
submission, needs no Metal event wait: queue order already places preceding work before
following commands. An unfenced, future, malformed, or already failed sync cannot be
satisfied by that queue. M3 stops replay at such a wait, marks the submission fence failed,
and makes `nvnSyncWait` report failure instead of silently succeeding or risking a wait on
work that could only be enqueued behind itself.

## Provenance and license

This layer is derived from the classic-Metal NVN backend of
[d2r-mobile](https://github.com/jhaizhou-ops/d2r-mobile) (a fork of nxnative):

- Upstream source: `d2r-mobile/dm/gfx/nvn/nvn_mtl.m` and `dm/gfx/nvn/nvn_copy_pitch.h`,
  commit `9393b00ea9aca52d9254214da93738f26582562f`
- Original author: jhaizhou
- License: MIT (below, as in d2r-mobile's LICENSE)

It was adapted to the current nxnative replay contract (object layouts, command records,
reflection, shader chain) with the OT2-specific and Route-A-specific machinery removed. The
integer-fragment-output, slice-geometry-folding and depth-mode entry handling follow upstream
nxnative ports (`nxnative/dm/gfx/nvn/nvn_mtl.m@0f99c353d3abc334c39855d691afdfab40f4639f`).
Individual hunks keep their origin notes in comments.

```
MIT License

Copyright (c) 2026 jhaizhou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
