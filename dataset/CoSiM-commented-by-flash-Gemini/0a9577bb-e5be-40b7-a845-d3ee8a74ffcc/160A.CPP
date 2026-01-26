#include <iostream>

using namespace std;

int main() {
  int total = 0, maximum = 0, money = 0, counter = 0;
  int flag = 0;
  int coins;
  cin >> coins;
  int value[coins];
  for (int i = 0; i < coins; i++) {
    cin >> value[i];
    total = total + value[i];
    if (value[i] > maximum) {
      maximum = value[i];
    }
  }
  while (flag == 0) {
    for (int i = 0; i < coins; i++) {
      if (maximum == value[i]) {
        money = money + value[i];
        counter++;
        if (money > (total / 2)) {
          cout << counter;
          flag = 1;
          break;
        }
      }
    }
    if (flag == 1) {
      break;
    }
    maximum--;
  }
  return 0;
}