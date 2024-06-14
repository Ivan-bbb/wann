# wann
Реализация алгоритма WANN на python.

Перед началом работы создайте виртуальное окружение и установите зависимости при помощи команды "poetry install" и активируйте его командой "poetry shell".

Для начала обучения поменяйте значение переменной task на нужную задачу в файле train.py и выполните команду "python train.py".\
Во время обучения сохраняются лучшие геномы в папке topology_genomes/{task}. Также сохраняются изображения тополгий сети в папке topology_images/{task}.\
Помимо этого в папке graphs сохраняется график зависимости лучшей и средней приспособленности от поколения.

Параметры алгоритма задаются в файле config.py.

Cancer:

![](https://github.com/Ivan-bbb/wann/blob/main/graphs/Cancer.png) 

Cart Pole:

![](https://github.com/Ivan-bbb/wann/blob/main/graphs/cartPole.png)

Для просмотра результатов обучения поменяйте значение переменной task на нужную задачу в файле test.py и выполните команду "python test.py".

Также можно сделать gif из сохраненных изображений, выполнив команду "python train.py". 

Cancer:

![](https://github.com/Ivan-bbb/wann/blob/main/videos/Cancer.gif) 

Cart Pole:

![](https://github.com/Ivan-bbb/wann/blob/main/videos/cartPole.gif)
