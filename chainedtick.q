\l lib/u.q

if[system "t";
  .z.ts:{
    .u.pub'[.u.t;value each .u.t];         / publish each table
    @[`.;.u.t;@[;`sym;`g#]0#]              / empty tables and apply grouped attribute to sym
  };
  upd:{[TABLE;DATA]
    TABLE insert DATA                      / insert DATA into TABLE
  }
  ];

if[not system "t";
  upd:.u.pub                               / overwrite upd with .u.pub
  ];

.u.rep:{[TABLE_SCHEMA]                     / .u.rep[,(table;schema)]
  (.[;();:;].) each TABLE_SCHEMA           / equivalent of`TABLE set TABLE
  };

.u.x:.z.x,(count .z.x)_enlist":5010"       / get the tickerplant port, default is 5010

/==============================================================

if[not system "p";                         / if port is not set
  system "p 5110"                          / listen on port 5110
  ];

.u.init .u.rep(.u.h:hopen`$":",.u.x 0)".u.sub[`;`]" / connect & subscribe to tickerplant for schema(s)