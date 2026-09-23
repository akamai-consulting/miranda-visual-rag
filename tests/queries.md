# Suggested Query Regression Set

| Query | Expected route / behavior |
|---|---|
| `black winter coat` | Direct English fashion → CLIP |
| `t-shirt with nerdy computer code` | Direct English fashion → CLIP |
| `Write shell code.` | Reject as out of scope |
| `Write a C++ Hello World program.` | Reject as out of scope |
| `Find me a T-shirt with shell code.` | Accept as fashion query |
| `冬に着る暖かいコートを見せて` | Multilingual → LLM normalization → CLIP |
| `Mostrami un cappotto nero per l'inverno.` | Multilingual → LLM normalization → CLIP |
