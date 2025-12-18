# r.q

in-memory realtime database

# Usage

    q r.q [tickerplant] [hdb]

## Parameters

 - `tickerplant`
 - `hdb`

**Examples**:

    # start rdb, connect to tickerplant and hdb for EOD
    $ q r.q :5010 :5012 -p 5013
    # start rdb, connect to chained tickerplant and hdb for EOD
    $ q r.q :5110 :5012 -p 5013

