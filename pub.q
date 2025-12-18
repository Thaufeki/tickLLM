\l schema/sym.q                            / load schemas

/ time timeExch sym exch bidpx bidqty bidtime askpx askqty asktime
levels:10;

sample:()!();
sample[`Book]:(.z.p;.z.p-1000000;`EURUSD;`HSFX;reverse pxs;qtys;times;1+pxs:"f"$100 + til levels;qtys:1e6*1+til levels;times:levels#.z.p);

.z.pc:{[HANDLE]
  if[HANDLE~.u.h;
    .u.h:0Ni
  ]
  };

.z.ts:{[]
  if[null .u.h;
    .u.h::@[hopen;.u.tp;0Ni]
  ];
  if[not null .u.h;
    t:first 1?tables[];
    if[t in key sample;
      .u.h (`.u.upd;t;sample[t]) // publish
    ]
  ];
  };

.u.h:0Ni; / tickerplant handle
.u.tp:`$first .z.x,enlist"::5010";         / default tickerplant port is localhost:5010

if[not system "t";
  system "t 1000"                          / publish every 1 second
  ];