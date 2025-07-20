#include <iostream>

using namespace std;
int money_to_borrow(int, int, int);

int main() {
  int first_banana_price, money, quantity;
  cin >> first_banana_price >> money >> quantity;
  cout << money_to_borrow(first_banana_price, money, quantity) << endl;
}

int money_to_borrow(int first_banana_price, int money, int quantity) {
  int money_needed = first_banana_price * (quantity * (quantity + 1) / 2);
  return (money_needed <= money) ? 0 : money_needed - money;
}