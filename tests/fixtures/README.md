# Synthetic ExpDec2 dataset

`synthetic_expdec2.csv` is deterministic and uses
`y = y0 + A_fast*exp(-x/t_fast) + A_slow*exp(-x/t_slow)` with these parameters:

| Series | y0 | A_fast | t_fast | A_slow | t_slow |
| --- | ---: | ---: | ---: | ---: | ---: |
| decay_a | 1.0 | 7.0 | 1.5 | 4.0 | 8.0 |
| decay_b | 0.8 | 5.0 | 1.0 | 3.2 | 6.0 |
| decay_c | 1.5 | 9.0 | 0.8 | 4.5 | 5.0 |

Each value has a checked-in fixed additive noise value with absolute magnitude at
most 0.02. `decay_b` has one empty value, `decay_c` has one `NaN`, and every Y has
a positive instrumental uncertainty column.
