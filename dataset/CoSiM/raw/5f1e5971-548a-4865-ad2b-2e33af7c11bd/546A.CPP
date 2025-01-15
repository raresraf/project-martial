#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int cost_of_banana, num_banana, have_money, need_money, need_to_borrow;
  cin >> cost_of_banana >> have_money >> num_banana;
  need_money = (num_banana * cost_of_banana * (num_banana + 1) / 2);
  if (need_money < have_money)
    need_to_borrow = 0;
  else
    need_to_borrow = need_money - have_money;

  cout << need_to_borrow;
}
