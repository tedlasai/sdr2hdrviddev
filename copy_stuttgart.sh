#!/usr/bin/env bash

BASE_URL="ftp://HdM-HDR-2014:Ht%eW84%p=@hdr-2014.hdm-stuttgart.de/HdM-HDR-2014_Original-HDR-Camera-Footage"
OUT_DIR="/data2/saikiran.tedla/hdrvideo/diff/data/stuttgart"

SCENES=(
  beerfest_lightshow_01
  beerfest_lightshow_02
  beerfest_lightshow_02_reconstruction_update_2015
  beerfest_lightshow_03
  beerfest_lightshow_04
  beerfest_lightshow_04_reconstruction_update_2015
  beerfest_lightshow_05
  beerfest_lightshow_06
  beerfest_lightshow_07
  bistro_01
  bistro_02
  bistro_03
  carousel_fireworks_01
  carousel_fireworks_04
  carousel_fireworks_05
  carousel_fireworks_06
  carousel_fireworks_07
  carousel_fireworks_08
  carousel_fireworks_09
  cars_closeshot
  cars_fullshot
  cars_longshot
  fireplace_01
  fireplace_02
  fishing_closeshot
  fishing_longshot
  hdr_testimage
  poker_fullshot
  poker_travelling_slowmotion
  showgirl_01
  showgirl_02
  smith_hammering
  smith_welding
)

for scene in "${SCENES[@]}"; do
  wget -r -nH --cut-dirs=1 \
    -P "${OUT_DIR}" \
    "${BASE_URL}/${scene}"
done
