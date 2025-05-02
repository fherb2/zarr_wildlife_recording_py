#!/bin/bash

fgrep -vf /workspace/.gitignore /workspace/.gitignore_suggestions > /tmp/gitignore.tmp
cat /tmp/gitignore.tmp >> /workspace/.gitignore
rm /tmp/gitignore.tmp