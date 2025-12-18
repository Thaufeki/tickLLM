\l lib/u.q                                 / load tick/u.q

\d .u                                      / enter .u namespace

ld:{[DATE]                                 / .u.ld[DATE]
  if[not type key L::` sv L,`$string DATE; / if logfile does not exist
    .[L;();:;()]                           / initialise to empty list
    ];
  if[0<=type i::j::-11!(-2;L);             / read log file
    -2 string[L]," is a corrupt log. Truncate to length ",string[last i]," and restart"; / write error to stderr
    exit 1                                 / exit with error code
    ];
  hopen L                                  / open handle to logfile and return it
  };

tick:{[SRC;DST]                            / .u.tick[path;logfile]
  init[];                                  / call .u.init[] (from tick/u.q)
  if[not all (min `time`sym in key flip value@) each t; / if tables do not have both a time and sym column
    '"timesym"                             / throw 'timesym exception
    ];
  @[;`sym;`g#]each t;                      / apply grouped attribute to sym column of each table
  d::.z.D;                                 / set .u.d to local date
  if[l::count DST;                         / if DST was specified
    L::`$":",DST,"/",SRC;                  / set .u.L as `:DST/SRC
    l::ld d                                / load/initialise the logfile
    ]
  };

endofday:{[]                               / .u.endofday[]
  end d;                                   / call .u.end for current date
  d+:1;                                    / increment date by one
  if[l;                                    / if we have an open handle to the logfile
    hclose l;                              / close the handle to the logfile
    l::0(`.u.ld;d)                         / initialise the new logfile
    ]
  };

ts:{[DATE]                                 / .u.ts[DATE]
  if[d<DATE;                               / if d is older than DATE (i.e. DATE is a new day)
    if[d<DATE-1;                           / if d is more than 1 day older than DATE
      system"t 0";                         / stop timer
      '"more than one day?"                / throw exception
      ];
    endofday[]                             / call .u.endofday
    ]
  };

// batch publishing
if[system "t";                             / if system timer is set
 .z.ts:{[]                                 / redefine .z.ts
    pub'[t;value each t];                  / call .u.pub for every subscriber
    @[`.;t;@[;`sym;`g#]0#];                / empty each table and then apply grouped attribute on sym
    i::j;                                  / update .u.i to include published messages
    ts .z.D                                / call .u.ts with current local date
    };
  upd:{[TABLE;DATA]                        / define .u.upd
    .debug.data:(DATA);
    if[not -12=type first first DATA;      / if the type of first item in DATA is not timestamp
      if[d<"d"$a:.z.P;                     / if local time, cast to date, is greater than .u.d
        .z.ts[]                            / call .z.ts
        ];
      DATA:$[0>type first DATA;            / if DATA is a single list
        a,DATA;                            / prepend timestamp to DATA
        enlist[(count first DATA)#a],DATA] / (otherwise list of lists), prepend length-DATA timestamps to DATA
      ];
    TABLE insert DATA;                     / insert DATA into TABLE
    if[l;                                  / if we have an open handle to the logfile
      l enlist (`upd;TABLE;DATA);          / write update to the logfile
      j+:1                                 / increment .u.j (total messages including buffer)
      ];
     }
   ];

// realtime publishing
if[not system "t";                         / if system timer is not set
  system "t 1000";                         / start timer (1 second resolution)
 .z.ts:{ts .z.D};                          / redefine .z.ts to call ts with current local date
 upd:{[TABLE;DATA]                         / redefine .u.upd
    .debug.data:DATA;
    ts "d"$a:.z.P;                         / call .u.ts with current local time
    if[not -12=type first first DATA;      / if the type of first item in DATA is not timestamp
      DATA:$[0>type first DATA;            / if DATA is a list
        a,DATA;                            / prepend timestamp to DATA
        enlist[(count first DATA)#a],DATA] / prepend length-DATA timestamps to DATA
      ];
   f:key flip value TABLE;                 / extract columns from TABLE
   pub[TABLE;$[0>type first DATA;          / if DATA is a single list
               enlist f!DATA;              / publish DATA as a table
               flip f!DATA]];              / (otherwise list of lists), publish as a table
   if[l;                                   / if we have an open handle to the logfile
     l enlist (`upd;TABLE;DATA);           / write update to the logfile
     i+:1                                  / increment .u.i (total messages)
     ];
   }
 ];

\d .                                       / leave .u namespace

/==============================================================

system "l schema/",(src:first .z.x,enlist"sym"),".q" / load schemas from sym.q

if[not system"p";                          / if port is not set
  system "p 5010"                          / open port 5010
  ];

.u.tick[src;.z.x 1];