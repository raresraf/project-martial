/**
 * @file 546A.CPP
 * @brief Calculates the amount of money to borrow for buying bananas.
 *
 * This program calculates the total cost of buying a certain number of bananas,
 * where the price of each banana increases linearly. It then determines how much
 * money needs to be borrowed if the initial amount is not sufficient.
 *
 * @b Algorithm: The total cost is calculated by summing an arithmetic progression.
 *    The cost of the i-th banana is `k * i`. The program iterates from 1 to `w` (the
 *    number of bananas) and accumulates the total cost. Finally, it subtracts the
 *    initial money (`n`) from the total cost to find the amount to borrow. If the
 *    initial money is enough, the amount to borrow is 0.
 *
 * @b Time_Complexity: O(W), where W is the number of bananas to be bought, due to the loop.
 * @b Space_Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int w, k, n, i;
  int total = 0;
  int pay;
  cin >> k >> n >> w; // k: cost of the first banana, n: initial money, w: number of bananas.

  /**
   * @brief This loop calculates the total cost of all bananas.
   * The cost of the i-th banana is k*i.
   */
  for (i = 1; i <= w; i++) {
    total = k * i + total;
  }
  
  // Note: The code prints the total cost first, which might be unintended.
  cout << total << endl;
  
  /**
   * @brief This block calculates the amount to be borrowed.
   * If the initial money 'n' is greater than the total cost, no money is needed.
   * Otherwise, it calculates the difference.
   */
  if (n > total) {
    cout << "0";
  } else {
    pay = total - n;
    cout << pay;
  }

  return 0;
}
