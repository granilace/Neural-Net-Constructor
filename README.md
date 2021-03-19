# Neural-Net-Constructor

Никита Свербягин, Виктор Куканов, Рубачёв Иван, Трошин Сергей

## Тестирование

```
brew install googletest / apt-get install gtest
cmake -S. -Bbuild 
cmake --build build
cd build
ctest --verbose
```
Для обновления тестов делаем `cmake --build build`