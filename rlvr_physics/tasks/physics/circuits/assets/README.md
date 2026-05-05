# Circuit Symbol Assets

This directory contains exported per-symbol schematic SVG assets derived from
Matthew Beckler's SVG Circuit Symbols sheet:

https://www.mbeckler.org/inkscape/circuit_symbols/

The source page says those drawings are released into the public domain. The
renderer loads the exported files directly and keeps placement metadata in
`symbol_assets.py`.

Small supplemental symbols that were missing from the sheet are drawn directly
as plain SVG files in the same normalized style.

Pin anchors may be declared in the SVG itself with hidden coordinate elements:

```xml
<g data-rlvr-role="pin-anchors" display="none">
  <circle id="pin-1" data-pin="1" cx="0" cy="24" r="0" />
</g>
```

The renderer uses these `data-pin` coordinates as the source of truth for
asset terminal locations.
