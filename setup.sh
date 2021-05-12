#!/bin/sh
### Download the flower data for the task 
mkdir flowers/ && wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz && tar -xvzf flower_data.tar.gz -C flowers
