# GBA Gymnasium gameplay smoke

Run 20261028_002105_827756Z restored the Seed/Red Mage native checkpoint and completed 24 discrete actions over 2 reproducible episodes.

- Reset observations reproducible: True
- Observation shape and dtype: [84, 84, 3] / uint8
- Neutral rewards on every action: True
- Episode result: terminated=False, truncated=True
- Stable-Baselines3 environment checker passed on this checkpoint. A CPU `CnnPolicy` was initialized and predicted/executed one action (`LEFT`); no training call ran (0 training steps).
- Source native checkpoint remained unchanged.
- Action counts: {'UP': 6, 'DOWN': 6, 'LEFT': 6, 'RIGHT': 6}
- Longest consecutive identical observation streak: 1 (diagnostic only; not a semantic stuck detector).

The contact sheet shows the Cornelia scene and response frames. This smoke validates the Gymnasium/core interface and input execution; it does not infer coordinates, battle outcomes, or story progress. No PPO training ran.
