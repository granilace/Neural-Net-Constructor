# Neural-Net-Constructor

Никита Свербягин, Виктор Куканов, Рубачёв Иван, Трошин Сергей

## Установка зависимостей

* macOS: `brew install googletest`
* Ubuntu: `apt-get install gtest`

## Тестирование

```
cmake -S. -Bbuild 
cmake --build build
cd build
ctest --verbose
```
Для обновления тестов делаем `cmake --build build`

## Как попробовать (CIFAR-10)

1. [Устанавливаем зависимости проекта](https://github.com/granilace/Neural-Net-Constructor#установка-зависимостей)
2. Получаем токен на странице `https://www.kaggle.com/<username>/account` (Create API Token) и кладём содержимое в `~/.kaggle/kaggle.json` 
3. Загружаем данные, собираем проект:
```
cd scripts
./prepare.sh
cd ..
cmake -S. -Bbuild 
cmake --build build
```
4. Запускаем:
```
cd build
./cifar
```
