\l lib/r.q                                 / load up common rdb codebase
\l lib/Q.q                                 / load up rewritten .Q functions

.u.end:{[DATE]                             / end of day: save, clear, hdb reload
  t:tables[];                              / get all tables
  t@:where `g=attr each t@\:`sym;          / extract tables that contain a grouped column
  .Q.hdpf[`$":",.u.x 1;`:.;DATE;`sym];     / write down tables and call HDB to reload from disk
  @[;`sym;`g#] each t;                     / apply grouped attribute to each table
  };

.u.x:.z.x,(count .z.x)_(":5010";":5012");  / get the tickerplant and history ports, defaults are 5010 and 5012 respectively

/==============================================================

if[not system"p";                          / if port is not set
  system "p 5011"                          / open port 5011
  ];

if[not "w"=first string .z.o;              / if not windows architecture
  system "sleep 1"                         / sleep for 1 second
  ];

.u.rep .(hopen `$":",.u.x 0)"(.u.sub[`;`];`.u `i`L)"; / connect to ticker plant for (schema;(logcount;log))