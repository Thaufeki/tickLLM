/ redefine par, qm, enx, en, dpft and hdpf in Q

\d .Q                                      / enter .Q namespace

par:{[DIR;PART;TABLE]                      / .Q.par[hdbpath;partition;table]
  ` sv (                                   / join filepath together with '/'
    $[type key h:` sv DIR,`par.txt;        / if par.txt exists
      `$":",h mod[PART;count h:read0 h];   / read par.txt and calculate next partition
       DIR];                               / else return original directory
    `$string PART;TABLE)                   / cast partition to symbol
  };

qm:{[x]
  $[(type x) or not count x;               / if atom, list or an empty list
    1;                                     / return 1
    t:type first x;                        / if atom or list
      min t=type each x;                   / return minimum type of each item (in case of nested mixed list)
      0]                                   / else mixed list, return 0
  };

enx:{[s;d;x]
  if[count f@:where {$[11h=type first x;min 11h=type each x;11h=type x]} each x f:key flip x; / if there are any symbol columns
    (` sv d,`sym)? distinct raze distinct each {$[0h=type x;raze x;x]} each x f / update sym file
  ];
  @[x;f;{$[0h=type y;(-1_ sums 0,count each y) cut x[`sym;raze y];x[`sym;y]]}s] / enumerate the column(s)
  };

en:enx[?;;];                               / .Q.en is .Q.enx with function as ? not $

dpft:{[D;P;F;T]
  if[not min qm each r:flip enx[$;D;value T]; / enx enumerates table, qm checks against mixed lists
    '"unmappable"                          / throw unmappable exception
    ];
  i:iasc T[F];                             / get sorted indices for parted column
  {[d;t;i;u;x]                             / {}[file;dictionary;sorted indices;attribute;column]
    @[d;x;:;u t[x] i]                      / sort each column by i, and apply #parted attribute to column F
    }[D:par[D;P;T];r;i;]'[(::;`p#) F=key r;key r]; / feed (::) or `p# attribute into sub-function
  @[D;`.d;:;F,r where not F=r:key r];      / create .d file
  T                                        / return table name
  };

hdpf:{[H;D;P;F]
  (@[`.;;0#]dpft[D;P;F;]@) each desc (count value @) each t:tables[]; / run dpft against each table in global namespace
  if[h:@[hopen;H;0];                       / if successful in opening handle to HDB process
    h (system;"l .");                      / send 'reload current directory' command to HDB
    hclose h                               / close handle
    ];
  };

\d .                                       / leave .Q namespace