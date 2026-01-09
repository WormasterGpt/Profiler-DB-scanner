# Profiler-DB-scanner
Profiler - The Python script for scanning DB using clues and finding info.
OPEN - SOURCE

How it works?

1.You import the DB to the database/sqlite/ or if you have unimportable to SQLite3 info, you can place the files to the database/Unimportable/

2.You starting the script and entering the clues for search

3.Script parses the DB(SQLite3) and unimportable files to find matches, and logs it to .log file in the profiler_log/

4.After scanning you can read the log file.

You can configure the script by editing the profiler.conf

Requirements:
Python 3.12 or newer, other in requirements file

How to launch?

python3 Profiler.app.py