/**
 * @file 546A.CPP
 * @brief This program calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of each subsequent banana increases linearly (1x, 2x, 3x, ...).
 * Algorithm: Linear summation of banana costs and comparison with initial money.
 * Time Complexity: O(N) where N is the number of bananas.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int costOfSingleBanana, totalCost = 0, numOfBanana, initialMoney,
                          moneyBorrowed;

  // Reads the cost of a single banana, the soldier's initial money, and the number of bananas to buy.
  cin >> costOfSingleBanana >> initialMoney >> numOfBanana;

  /**
   * Block Logic: Calculates the total cost for buying 'numOfBanana' bananas.
   * The cost of the i-th banana is i * costOfSingleBanana.
   * Invariant: 'totalCost' accumulates the sum of costs for bananas from 1 to 'i'.
   */
  for (int i = 1; i <= numOfBanana; i++) {
    totalCost = totalCost + (i * costOfSingleBanana);
  }
  /**
   * Block Logic: Determines if money needs to be borrowed and calculates the amount.
   * If 'totalCost' exceeds 'initialMoney', the difference is the amount to be borrowed.
   * Otherwise, no money needs to be borrowed.
   */
  if (totalCost >= initialMoney) {
    moneyBorrowed = totalCost - initialMoney;
    cout << moneyBorrowed << endl;
  } else
    cout << 0 << endl;

  // Standard program exit.
  return 0;
}