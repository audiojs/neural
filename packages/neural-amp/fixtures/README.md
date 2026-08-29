# Test fixtures

- `wavenet_a1_standard.nam`: MIT-licensed example model (random weights) from [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore/tree/main/example_models). Committed; used for format/parse tests.
- `5150.nam`: real trained capture (Peavey 5150 Block Letter, boosted) from the [pelennor2170/NAM_models](https://github.com/pelennor2170/NAM_models) community collection. No explicit license, so it is fetched, never committed (per the umbrella weights policy). Behavior tests skip when absent. Fetch:

```sh
curl -o 5150.nam 'https://raw.githubusercontent.com/pelennor2170/NAM_models/main/Helga%20B%205150%20BlockLetter%20-%20Boosted.nam'
```
