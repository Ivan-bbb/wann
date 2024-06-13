# wann
Реализация алгоритма WANN на python.

Параметры алгоритма задаются в файле config.py.

Для начала обучения поменяйте значение переменной task на нужную задачу в файле train.py и выполните команду "python train.py". 
Во время обучения сохраняются лучшие геномы в папке topology_genomes/{task}. Также сохраняются изображения тополгий сети в папке topology_images/{task}.
Помимо этого в папке graphs сохраняется график зависимости лучшей и средней приспособленности от поколения.

Cancer:

![](https://github.com/Ivan-bbb/wann/blob/main/graphs/cancer.png) 

Cart Pole:

![](https://github.com/Ivan-bbb/wann/blob/main/graphs/cartPole.png)

Для просмотра результатов обучения поменяйте значение переменной task на нужную задачу в файле test.py и выполните команду "python test.py".

Также можно сделать gif из сохраненных изображений, выполнив команду "python train.py". 

Cancer:

![](https://github.com/Ivan-bbb/wann/blob/main/videos/cancer.gif) 

Cart Pole:

![](https://github.com/Ivan-bbb/wann/blob/main/videos/cartPole.gif)
