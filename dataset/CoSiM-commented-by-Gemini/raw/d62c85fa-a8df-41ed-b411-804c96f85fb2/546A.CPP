/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A, "Soldier and Bananas".
 * @details This program calculates the amount of money a soldier needs to borrow.
 * The cost of the bananas increases linearly. The program calculates the total
 * cost and subtracts the soldier's initial money to find the deficit.
 *
 * @algorithm Iterative Summation
 * @complexity
 * Time: O(W), where W is the number of bananas to buy. The complexity is determined
 *       by the loop to calculate the total cost.
 * Space: O(1), as it uses a fixed number of variables.
 */
#include <iostream>
using namespace std;

int main() {
  // Use long to prevent potential integer overflow when calculating the total price,
  // as w can be up to 1000 and k up to 1000.
  long k, n, w; 
  // k: cost of the first banana
  // n: initial dollars the soldier has
  // w: number of bananas he wants to buy
  cin >> k >> n >> w;
  
  long price = 0; // Variable to accumulate the total cost.

  /**
   * @brief Loop to calculate the total price of 'w' bananas.
   * The cost of the i-th banana is i * k.
   */
  for (int i = 1; i <= w; i++) {
    price += (i * k);
  }

  /**
   * @brief Determine the amount to borrow.
   * If the soldier's money 'n' is enough to cover the price, he borrows 0.
   * Otherwise, he borrows the difference.
   */
  if (n >= price) {
    cout << "0" << endl;
  } else {
    cout << price - n << endl;
  }

  return 0; // Indicate successful execution.
}