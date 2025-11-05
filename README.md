IKT457 - hex game using GTM

To look at Optuna study: run optuna-dashboard sqlite:///optuna_study.db --host 127.0.0.1 --port 8081

TODO: If the games are generated randomly, maybe implement some sort of smart placements, so we get better data.

TODO: Not only train/test split, but we should do the train/val/test split, so we dont optimize on test split.

TODO: Add meta data so we can see on what games the potentially fail (0, -2, or -5)

TODO: Add some additional inputs

TODO: Automatically generate symbols via pattern mining or attention over adjacency graphs, Right now it's hand-engineered in parse_games.py