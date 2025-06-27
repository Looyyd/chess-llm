The goal of this project was to create a large language model reasoning finetune on chess games.

At first glance, this seems like a good fit for Reinforcment Learning From Verifiable Reward(RLVR), because for each move the stockfish evaluation change can be used as the reward, which allows the RL process to be done move by move, and not with a game by game sparse reward.

However, here are some issues encountered when attempting this:
- The base models are terrible at chess, this goes for both open source models such as Qwen 7B, but also for the SOTA models such as Claude 4 Opus with reasoning.
- Therefore, if used in RL with reasoning, the based models give non sensical reasoning traces, and so there is not benefit gained from using reasoning.

To try to help with this issue, the script train_sft.py finetunes Qwen 7B on a sort of "pretraining datasetset" to help the base model be better at chess. More speciffically it uses a prompt/completion format where the user prompt shows a chess game(from the Lichess.com dataset of played games) and asks the assistant to predict the next move played.
However, there could be issues like catastrophic forgetting with this method, ideally the chess data should be included in the pretraining set of the base model, and not applied on the Assistant tuned model.

Still this was attempted. And it does seems to kinda work, because the loss goes down smoothly and after some training the model does propose sensible moves!
So now the model has some understanding of the chessboard, but it still can't reason over it.

To solve this second issue, a dataset of LLM generated reasoning on chess games was created, and finetuned on in train_format_alignment.py.

As said before, the LLM generated reasoning is bad, but still we hope that it's might be good enough to "kickstart" the RL process, if at least some traces are good.

However, after finetuning on this dataset, and then using the reasoning format aligned model in a GRPO loop(using train_rl.py), the model still doesn't perform. It shows a behaviour where the reasoning remains non sensical, and the final moves get slightly better with time. So this is again not what we want.

To try to make this better, a script was created to try to create a high quality dataset of chess reasoning. The dataset was created by recording a human voice analysing the position and then using the openai transcription api to change it into text.
120 examples were created, which is a low amount but it does take around 2/3 minutes per sample to create the full reasoning.
However 120 is not enough, as the training process overfits.

Because of all these complications, the projects is paused for now. It might just be easier to wait for better base models.