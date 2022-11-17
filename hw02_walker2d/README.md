## HW02: Walker2D with Policy Gradient

### Task
В данном задании необходимо обучить агента побеждать в игре Walker2D при помощи Actor-Critic, A2C, TRPO или PPO. Для решения задачи можно трансформировать состояние и награду среды.
К заданию также нужно приложить код обучения агента (не забудьте зафиксировать seed!), готовый (уже обученный) агент должен быть описан в классе Agent в файле 'agent.py'.
Оценка выставляется от 1 до 10 и линейно зависит за набранный агентом счет в среднем за 50 эпизодов. Максимальный счет - 1200, минимальный счет - 200.
Обратите внимание: для выполнения данной работы вам потребуется установить библиотеку PyBullet.

#### До обучения:
![Before](https://github.com/AnnaSmelova/RL/blob/main/hw02_walker2d/video/Walker2DBulletEnv-v0_first.gif)

#### После обучения:
![After](https://github.com/AnnaSmelova/RL/blob/main/hw02_walker2d/video/Walker2DBulletEnv-v0_best.gif)
