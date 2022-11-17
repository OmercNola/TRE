#!/bin/bash

nvidia-smi
ps -up `nvidia-smi |tail -n +16 | head -n -1 | sed 's/\s\s*/ /g' | cut -d' ' -f3` 
