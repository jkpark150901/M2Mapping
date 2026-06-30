enroot start --root --rw \
  --mount /home/dev/jkpark/M2Mapping:/m2mapping_ws/src/M2Mapping \
  --mount /home/dev/jkpark/datasets:/datasets \
  m2mapping



#   enroot start --root --rw \
#   --mount /home/dev/jkpark/M2Mapping:/m2mapping_ws/src/M2Mapping \
#   --mount /home/dev/jkpark/datasets:/datasets \
#   --mount /home/dev/jkpark/FAST-LIVO2:/m2mapping_ws/src/FAST-LIVO2 \
#   --mount /home/dev/jkpark/rpg_vikit:/m2mapping_ws/src/rpg_vikit \
#   --mount /home/dev/jkpark/Sophus:/Sophus \
#   --mount /tmp/.X11-unix:/tmp/.X11-unix \
#   --env DISPLAY=$DISPLAY \
#   m2mapping