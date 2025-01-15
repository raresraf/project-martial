#include <bits/stdc++.h>
using namespace std;

int main() {
  int costOfSingleBanana, totalCost = 0, numOfBanana, initialMoney,
                          moneyBorrowed;

  cin >> costOfSingleBanana >> initialMoney >> numOfBanana;

  for (int i = 1; i <= numOfBanana; i++) {
    totalCost = totalCost + (i * costOfSingleBanana);
  }
  if (totalCost >= initialMoney) {
    moneyBorrowed = totalCost - initialMoney;
    cout << moneyBorrowed << endl;
  } else
    cout << 0 << endl;

  return 0;
}