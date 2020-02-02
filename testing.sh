#!/usr/bin/env bash
for i in cut*
	do echo -e "\n\n${i}:\n"
	python3.7 drivingplate.py ${i}
done
echo -e "\n"
