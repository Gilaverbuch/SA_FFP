#!/bin/bash


export xsize='4i'
export ysize='4i'
export y2='0.02'
export y1='-0.02'
export x2='1'
export x1='0.5'
export filename='TL_contour.ps'
gmt set PS_PAGE_ORIENTATION portrait
gmt set MAP_LABEL_OFFSET 0.3c
gmt set ANNOT_FONT_PRIMARY 12

paste  xyz_high_res.txt| awk '{print $1,$2,$3}' | gmt pscontour  -R${x1}/${x2}/${y1}/${y2} -B0.1:"kz":/0.005:"dc/dz":WSne -JX${xsize}/${ysize} -C10 -A30+a30+s7 -Gd1.5i -W0.5p > ${filename}

ps2eps -B  -f ${filename}
epstopdf  ${filename%???}.eps