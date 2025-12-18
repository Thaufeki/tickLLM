/ common rdb code

upd:insert                                 / replace upd with insert

.u.rep:{[TABLE_SCHEMA;COUNT_LOG]           / .u.rep[,(table;schema);(logcount;log)]
  (.[;();:;].) each TABLE_SCHEMA;          / initialise each table with schema
  if[null first COUNT_LOG;                 / if null logcount
    :()                                    / return (nothing to replay)
    ];
  -11!COUNT_LOG                            / replay logfile -11!(logcount;log)
  };