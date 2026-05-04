graph [
  directed 1
  node [
    id 0
    label "AKT"
  ]
  node [
    id 1
    label "MEK1"
  ]
  node [
    id 2
    label "ERK12"
  ]
  node [
    id 3
    label "JNK"
  ]
  node [
    id 4
    label "IKB"
  ]
  node [
    id 5
    label "p38"
  ]
  node [
    id 6
    label "HSP27"
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 0
    target 4
  ]
  edge [
    source 1
    target 2
  ]
  edge [
    source 1
    target 3
  ]
  edge [
    source 2
    target 6
  ]
  edge [
    source 3
    target 4
  ]
  edge [
    source 4
    target 6
  ]
  edge [
    source 5
    target 3
  ]
  edge [
    source 5
    target 6
  ]
]
