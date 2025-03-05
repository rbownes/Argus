#!/usr/bin/env python3
"""
Simple script to make the run_tests.sh file executable.
This is a workaround since we can't directly run chmod.
"""
import os
import stat

script_path = 'run_tests.sh'
current_perms = os.stat(script_path).st_mode
os.chmod(script_path, current_perms | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
print(f"Made {script_path} executable")
