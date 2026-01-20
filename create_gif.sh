#!/bin/bash
# Create GIF from frames in frames_20260120_113139
ffmpeg -r 5 -i frames_20260120_113139/frame_%04d.png -vf palettegen palette.png
ffmpeg -r 5 -i frames_20260120_113139/frame_%04d.png -i palette.png -lavfi paletteuse sporozoite_simulation_20260120_113144.gif
rm palette.png
