# Gate fixtures

Committed eval parquets for CI. Regenerated from local `runs/` after
threshold rewrite `eval_id=Bp`, `bounds=recorded+15%`.

| File | Meaning | Expected `robogate gate` |
| --- | --- | --- |
| `baseline.parquet` | B' (`act-coffee-30000`) on `scenarios/real` | — |
| `candidate.parquet` | Current release candidate: same B' eval | green vs baseline |
| `known-bad/published-b.parquet` | `gozdebaydogmus/act-aloha-static-coffee-test@646846823c8473712f689f59bdd03f798a688f6a` | red vs baseline |

## Regenerating

```bash
robogate eval scenarios/real --runs runs --version-contains act-coffee-30000 --eval-id Bp-calib
robogate eval scenarios/real --runs runs --version-contains act-aloha-static-coffee-test --eval-id B-published
cp results/Bp-calib.parquet fixtures/gate/baseline.parquet
cp results/Bp-calib.parquet fixtures/gate/candidate.parquet
cp results/B-published.parquet fixtures/gate/known-bad/published-b.parquet
```

Changing any `scenarios/real/*.yaml` field that enters `content_hash` without
regenerating these files makes the green gate fail (hash mismatch).

## Runs used

| scenario | B' | published B |
| --- | --- | --- |
| aloha-static-coffee-ep0 | `…act-coffee-30000__20260914T144519Z` | `…act-aloha-static-coffee-test__20260914T155054Z` |
| aloha-static-coffee-ep1 | `…20260914T144803Z` | `…20260914T155243Z` |
| aloha-static-coffee-ep2 | `…20260914T145047Z` | `…20260914T155441Z` |
| aloha-static-coffee-ep3 | `…20260914T145341Z` | `…20260914T155630Z` |
| aloha-static-coffee-ep4 | `…20260914T145559Z` | `…20260914T155822Z` |
| aloha-static-coffee-ep5 | `…20260914T145812Z` | `…20260914T160021Z` |
| aloha-static-coffee-ep6 | `…20260914T150128Z` | `…20260914T160217Z` |
| aloha-static-coffee-ep7 | `…20260914T150355Z` | `…20260914T160407Z` |
| aloha-static-coffee-ep8 | `…20260914T150619Z` | `…20260914T160606Z` |
| aloha-static-coffee-ep9 | `…20260914T150857Z` | `…20260914T160750Z` |
| coffee-published-action-space-mismatch | `…act-coffee-30000__20260915T070515Z` | `…act-aloha-static-coffee-test__20260915T070609Z` |

B' checkpoint: `/root/autodl-tmp/ckpts/v2/act-coffee-30000` (frozen bench fixture, not retrained).
Intended Hub copy: `davore/act-aloha-static-coffee-bprime` (private). Upload with
`HF_TOKEN=… python scripts/remote/upload_bprime.py`. Local backup:
`.cache/bprime/act-coffee-30000` (gitignored). Do not point
`scenarios/real/*.yaml` `target.checkpoint` at the Hub id — that enters
`content_hash` and would invalidate these fixtures.
