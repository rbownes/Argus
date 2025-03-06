#!/bin/bash

# find-python-imports.sh
# Script to find all imported libraries in a Python project

# Set the project directory, default to current directory if not provided
PROJECT_DIR="${1:-.}"

# List of standard library packages to exclude
STDLIB_PACKAGES=(
    "abc" "aifc" "argparse" "array" "ast" "asyncio" "atexit" "audioop" "base64" "bdb"
    "binascii" "binhex" "bisect" "builtins" "bz2" "calendar" "cgi" "cgitb" "chunk" "cmath"
    "cmd" "code" "codecs" "codeop" "collections" "colorsys" "compileall" "concurrent" "configparser"
    "contextlib" "copy" "copyreg" "cProfile" "crypt" "csv" "ctypes" "curses" "dataclasses"
    "datetime" "dbm" "decimal" "difflib" "dis" "distutils" "doctest" "dummy_threading" "email"
    "encodings" "ensurepip" "enum" "errno" "faulthandler" "fcntl" "filecmp" "fileinput" "fnmatch"
    "formatter" "fractions" "ftplib" "functools" "gc" "getopt" "getpass" "gettext" "glob"
    "grp" "gzip" "hashlib" "heapq" "hmac" "html" "http" "idlelib" "imaplib" "imghdr"
    "imp" "importlib" "inspect" "io" "ipaddress" "itertools" "json" "keyword" "lib2to3"
    "linecache" "locale" "logging" "lzma" "macpath" "mailbox" "mailcap" "marshal" "math"
    "mimetypes" "mmap" "modulefinder" "msilib" "msvcrt" "multiprocessing" "netrc" "nis"
    "nntplib" "numbers" "operator" "optparse" "os" "ossaudiodev" "parser" "pathlib" "pdb"
    "pickle" "pickletools" "pipes" "pkgutil" "platform" "plistlib" "poplib" "posix" "pprint"
    "profile" "pstats" "pty" "pwd" "py_compile" "pyclbr" "pydoc" "queue" "quopri" "random"
    "re" "readline" "reprlib" "resource" "rlcompleter" "runpy" "sched" "secrets" "select"
    "selectors" "shelve" "shlex" "shutil" "signal" "site" "smtpd" "smtplib" "sndhdr" "socket"
    "socketserver" "spwd" "sqlite3" "ssl" "stat" "statistics" "string" "stringprep" "struct"
    "subprocess" "sunau" "symbol" "symtable" "sys" "sysconfig" "syslog" "tabnanny" "tarfile"
    "telnetlib" "tempfile" "termios" "test" "textwrap" "threading" "time" "timeit" "tkinter"
    "token" "tokenize" "trace" "traceback" "tracemalloc" "tty" "turtle" "turtledemo" "types"
    "typing" "unicodedata" "unittest" "urllib" "uu" "uuid" "venv" "warnings" "wave" "weakref"
    "webbrowser" "winreg" "winsound" "wsgiref" "xdrlib" "xml" "xmlrpc" "zipapp" "zipfile"
    "zipimport" "zlib" "_thread" "__future__"
)

echo "Scanning Python files in $PROJECT_DIR for imports..."

# Create a temporary file for processing
TEMP_FILE=$(mktemp)

# Find all Python files and extract import statements
find "$PROJECT_DIR" -type f -name "*.py" | while read -r file; do
    # Extract lines that start with "import" or "from"
    grep -E "^[[:space:]]*(import|from)[[:space:]]+" "$file" >> "$TEMP_FILE"
done

echo -e "\nProcessing imports..."

# Process the imports to extract library names
cat "$TEMP_FILE" | while read -r line; do
    # Process "import X" or "import X as Y" or "import X, Y, Z"
    if [[ $line =~ ^[[:space:]]*import[[:space:]]+(.+) ]]; then
        imports="${BASH_REMATCH[1]}"
        # Split by commas for multiple imports in one line
        echo "$imports" | tr ',' '\n' | while read -r imp; do
            # Extract the actual package name (before "as" if present)
            if [[ $imp =~ ^[[:space:]]*([a-zA-Z0-9_\.]+) ]]; then
                base_package="${BASH_REMATCH[1]}"
                # Get the top-level package by splitting on dots
                echo "$base_package" | cut -d. -f1
            fi
        done
    # Process "from X import Y" statements
    elif [[ $line =~ ^[[:space:]]*from[[:space:]]+([a-zA-Z0-9_\.]+)[[:space:]]+import ]]; then
        base_package="${BASH_REMATCH[1]}"
        # Get the top-level package
        echo "$base_package" | cut -d. -f1
    fi
done | sort | uniq > "$TEMP_FILE.sorted"

# Count the number of unique libraries
LIB_COUNT=$(wc -l < "$TEMP_FILE.sorted")

echo -e "\nFound $LIB_COUNT unique Python libraries:\n"

# Display the final list
cat "$TEMP_FILE.sorted"

# Filter out standard library packages
echo -e "\nFiltering out standard library packages..."
cat "$TEMP_FILE.sorted" | while read -r pkg; do
    is_stdlib=0
    for stdlib in "${STDLIB_PACKAGES[@]}"; do
        if [[ "$pkg" == "$stdlib" ]]; then
            is_stdlib=1
            break
        fi
    done
    if [[ $is_stdlib -eq 0 ]]; then
        echo "$pkg" >> "$TEMP_FILE.filtered"
    fi
done

# Count the number of third-party libraries after filtering
if [[ -f "$TEMP_FILE.filtered" ]]; then
    FILTERED_COUNT=$(wc -l < "$TEMP_FILE.filtered")
    echo -e "\nFound $FILTERED_COUNT third-party Python libraries (after removing standard library):\n"
    
    # Display the filtered list
    cat "$TEMP_FILE.filtered"
    
    # Create the uv add command for filtered packages
    UV_ADD_CMD="uv add"
    while read -r lib; do
        UV_ADD_CMD="$UV_ADD_CMD $lib"
    done < "$TEMP_FILE.filtered"
    
    echo -e "\nUV add command:"
    echo "$UV_ADD_CMD"
else
    echo -e "\nNo third-party packages found after filtering standard library packages."
    UV_ADD_CMD="uv add"
    echo -e "\nUV add command (empty):"
    echo "$UV_ADD_CMD"
fi

# Clean up temporary files
rm "$TEMP_FILE" "$TEMP_FILE.sorted"
if [[ -f "$TEMP_FILE.filtered" ]]; then
    rm "$TEMP_FILE.filtered"
fi

echo -e "\nDone!"