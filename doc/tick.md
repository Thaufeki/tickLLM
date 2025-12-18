# tick.q



# Usage

    q tick.q <schema> [logpath] [-p PORT]

 - `-p`, port to listen on, defaults to 5010 if not specified
 - `<schema>`, name of .q script containing table schemas, as well as base folder for logfile
 - `[logpath]`, directory that logfile should written to

**Examples**:

    # load up ./tick/sym.q for schemas, writes logfile to /data/log/sym/YYYY.MM.DD
    $ q tick.q sym /data/log
    # load up ./tick/schemas.q for schemas, do not write logfile
    $ q tick.q schemas

Note that `tick.q` has two modes for publishing, batched and realtime.

## Batched Operation (publish updates on a timer)

    $ q tick.q sym . -p 5010 -t 1000

 - When `.u.upd[TABLE;DATA]` is called, DATA is inserted into TABLE.
 - `.z.ts` triggers publishing of each table in `t` to subscribers, then tables are emptied
 - If logging is enabled, the update is written to the logfile

## Real-time Operation (publish updates as soon as they are received)

    $ q tick.q sym . -p 5010 -t 1000

 - When `.u.upd[TABLE;DATA]` is called, DATA is pushed to all subscribers
 - If logging is enabled, the update is written to the logfile

# Development


## Global variables

```
 .u.subs - dictionary of tables->(handle;syms)
 .u.i    - msg count in log file
 .u.j    - total msg count (log file plus those held in buffer)
 .u.t    - table names
 .u.L    - tp log filename, e.g. `:./sym2008.09.11
 .u.l    - handle to tp log file
 .u.d    - date
```

## Assumptions

 - `upd` expects `DATA` to be a list or list-of lists (not dictionaries or table rows)
 - all table schemas should be known at start-up (e.g. defined in `tick/sym.q` file)