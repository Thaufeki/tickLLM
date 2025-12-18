/ common tick code

\d .u

init:{[]                                   / .u.init[]
  subs::t!(count t::tables[])#()           / initialise subscribers dictionary
  };

del:{[TABLE;HANDLE]                        / .u.del[TABLE;HANDLE]
  subs[TABLE]_:subs[TABLE;;0]?HANDLE       / drop HANDLE from subscriptions for TABLE
  };

.z.pc:{[HANDLE]                            / socket close callback
  del[;HANDLE] each t                      / remove subscription for HANDLE from all tables
  };

sel:{[DATA;SYMS]
  $[`~SYMS;                                / if SYMS is null
    DATA;                                  / return DATA
    select from DATA where sym in SYMS]    / return filtered DATA
  };

pub:{[TABLE;DATA]
  {[T;X;W]
    if[count X:sel[X;W 1];                 / if there is (filtered) DATA to publish
      (neg first W)(`upd;T;X)              / async publish of (`upd;TABLE;DATA)
      ]
    }[TABLE;DATA;] each subs[TABLE]        / call lambda for each subscriber for TABLE
  };

add:{[TABLE;SYMS]
  $[(count subs[TABLE])>i:subs[TABLE;;0]?.z.w; / if z.w is already a subscriber
    .[`.u.subs;(TABLE;i;1);union;SYMS];    / add new subs for existing subscriber
    subs[TABLE],:enlist (.z.w;SYMS)        / (new subscriber), append to subs
    ];
  (TABLE;$[99=type v:value TABLE;          / return (TABLE;...) where ... is either
       sel[v;SYMS];                        / the empty ditionary for TABLE
       0#v])                               / the empty table of TABLE
  };

sub:{[TABLE;SYMS]                          / .u.sub[TABLE;SYMS], called by client to subscribe to table(s) and syms
  if[TABLE~`;                              / if TABLE is null
    :sub[;SYMS] each t                     / subscribe to all tables
    ];
  if[not TABLE in t;                       / if TABLE is not in list of available tables
    'TABLE                                 / throw exception
    ];
  del[TABLE;.z.w];                         / remove any existing subscriptions for subscriber
  add[TABLE;SYMS]                          / add subscription for table
  };

end:{[DATE]                                / .u.end[DATE], notifies subscribers of end of day
  (neg (union/) subs[;;0])@\:(`.u.end;DATE) / async send (`.u.end;DATE) to each subscriber
  };

\d .