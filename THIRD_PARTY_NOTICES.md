# Third-party notices

The vendored models retain their MIT license files:

- ASFormer: https://github.com/ChinaYi/ASFormer — pinned author commit e1bbe4f3ed083748f91467c51a63ac2a8b9277ad. License: experiments/hospital_temporal_compare/vendor/LICENSE_asformer.txt.
- MS-TCN2 / MS-TCN++: https://github.com/sj-li/MS-TCN2 — pinned author commit f423a9e65f4ccb1cd7322eb9f94946a19e787993. License: experiments/hospital_temporal_compare/vendor/LICENSE_mstcn2.txt.

The vendor modules are the project's accepted clip-level adaptations, not claimed verbatim upstream files or a full frame-level official reproduction. ASFormer includes per-segment/same-length handling and a short-window computation preserving padding softmax mass; MS-TCN++ includes native clip-level use and same-length grouping in the adapter. See published source for the exact implementation.

Uni-AdaFocus adapter code references an external original source tree and public official SSv2 weights. Those external source files and weights are not bundled or relicensed by this snapshot. The adapter alone is not a complete Uni-AdaFocus distribution. No private FSN training weights are included.
