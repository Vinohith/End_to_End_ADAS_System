#!/bin/sh
python spawn_npc.py -n 50 &
python manual_control_1.py &
python grab_screen_lane_marking_road_Cross_obj.py &
sleep 30s
python detect_drowsiness.py