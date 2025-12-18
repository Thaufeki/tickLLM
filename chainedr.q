\l lib/r.q                                 / load up common rdb codebase

.u.end:{
  @[`.;tables[];@[;`sym;`g#]0#];           / empty each table and then apply grouped attribute on sym
  };

.u.x:.z.x,(count .z.x)_enlist":5110"       / get the chained tickerplant port, default is 5110

/==============================================================

if[not system"p";                          / if port is not set
  system "p 5111"                          / open port 5111
  ];

if[not "w"=first string .z.o;              / if not windows architecture
  system "sleep 1"                         / sleep for 1 second
  ];

.u.rep .(hopen`$":",.u.x 0)"(.u.sub[`;`];$[`h in key `.u;.u.h(`.u;`i`L);.u[`i`L]])" / fetch (schema;(logcount;log)) from tickerplant (potentially via chained ticker plant)