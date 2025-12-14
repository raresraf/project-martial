#include <iostream>

using namespace std;

int main() {
  int cost, money, bananas, borrow = 0;
  cin >> cost >> money >> bananas;
  for (int i = 1; i <= bananas; i++) {
    borrow = borrow + (cost * i);
  }
  if (borrow > money) {
    borrow = borrow - money;
    cout << borrow;
  } else {
    cout << '0';
  }
  return 0;
}