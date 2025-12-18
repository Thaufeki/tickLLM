# chainedtick.q

As tick.q but does not write a logfile.

# Usage

    $ q chainedtick.q <tickerplant> [-p PORT]

 - `<tickerplant>`, tickerplant to connect to, defaults to :5010 if not specified
 - `-p`, port to listen on, defaults to 5110 if not specified

**Examples:**

    # connect to local tp on port 5010, publish updates immediately, listen on port 5110
    $ q chainedtick.q :5010 -p 5110
    # connect to local tp on port 5010, publish batched updates every second, listen on port 5110
    $ q chainedtick.q :5010 -p 5110 -t 1000

# Development

## Global variables

```
 .u.h - handle to tickerplant
```