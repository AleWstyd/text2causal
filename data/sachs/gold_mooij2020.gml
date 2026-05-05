graph [
  comment "Source: Mooij, J.M., Magliacane, S. and Claassen, T. (2020). Joint Causal Inference from Multiple Contexts. JMLR 21(99), 1-68. Sec. 5.8 (Sachs flow cytometry), Fig. 39-40 and p. 77 bootstrap summary: ancestral relations Mek to Raf, PLCg to PIP2, Akt to Erk, P38 to PKC vs. Sachs et al. (2005) consensus DAG."
  comment "Directed acyclic transcription: from CDT Sachs consensus, replace Raf to Mek with Mek to Raf, PKC to P38 with P38 to PKC, add Akt to Erk alongside Mek to Erk. Node labels match dataset columns."
  comment "PDF accessed: https://jmlr.org/papers/volume21/17-123/17-123.pdf (access date 2026-05-05)."
  directed 1
  node [
    id 0
    label "P38"
  ]
  node [
    id 1
    label "PIP2"
  ]
  node [
    id 2
    label "PIP3"
  ]
  node [
    id 3
    label "PKA"
  ]
  node [
    id 4
    label "PKC"
  ]
  node [
    id 5
    label "p44/42"
  ]
  node [
    id 6
    label "pjnk"
  ]
  node [
    id 7
    label "pakts473"
  ]
  node [
    id 8
    label "plcg"
  ]
  node [
    id 9
    label "pmek"
  ]
  node [
    id 10
    label "praf"
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
    target 4
  ]
  edge [
    source 2
    target 7
  ]
  edge [
    source 2
    target 8
  ]
  edge [
    source 3
    target 0
  ]
  edge [
    source 3
    target 5
  ]
  edge [
    source 3
    target 6
  ]
  edge [
    source 3
    target 7
  ]
  edge [
    source 3
    target 9
  ]
  edge [
    source 3
    target 10
  ]
  edge [
    source 4
    target 6
  ]
  edge [
    source 4
    target 9
  ]
  edge [
    source 4
    target 10
  ]
  edge [
    source 7
    target 5
  ]
  edge [
    source 8
    target 1
  ]
  edge [
    source 8
    target 4
  ]
  edge [
    source 9
    target 5
  ]
  edge [
    source 9
    target 10
  ]
]
