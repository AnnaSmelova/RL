## HW03: Ant with Deterministic Policy Gradient

### Task
В данном задании необходимо обучить агента побеждать в игре Ant при помощи DDPG, TD3 или SAC. Для решения задачи можно трансформировать состояние и награду среды.<br>
К заданию также нужно приложить код обучения агента (не забудьте зафиксировать seed!), готовый (уже обученный) агент должен быть описан в классе Agent в файле 'agent.py'.<br>
Оценка выставляется от 1 до 10 и линейно зависит за набранный агентом счет в среднем за 50 эпизодов. Максимальный счет - 2200, минимальный счет - 1000.<br>
Обратите внимание: для выполнения данной работы вам потребуется установить библиотеку PyBullet.

#### До обучения:
![Before](https://github.com/AnnaSmelova/RL/blob/main/hw03_ant/video/AntBulletEnv-v0_first.gif)

#### После обучения:
![After](https://github.com/AnnaSmelova/RL/blob/main/hw03_ant/video/AntBulletEnv-v0_best.gif)
