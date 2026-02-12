/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas. The cost of the bananas increases linearly: the first costs `k`,
 * the second `2k`, and so on. The program iteratively calculates the total cost and
 * subtracts the soldier's initial money to find the deficit.
 *
 * @algorithm Iterative summation of an arithmetic series.
 * @complexity
 * Time complexity: O(W), where W is the number of bananas, due to the loop.
 * Space complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // f: the cost of the first banana (k).
  int f;
  // init: the initial amount of money the soldier has (n).
  int init;
  // bananas: the total number of bananas to buy (w).
  int bananas;

  // ret: the running total cost of the bananas.
  int ret = 0;
  cin >> f >> init >> bananas;
  
  // eve: stores the base cost of the first banana to be added in each step.
  int eve = f;
  
  // Handle edge cases where no cost is incurred.
  if (bananas == 0 || f == 0) {
    cout << 0;
    return 0;
  }
  
  /**
   * @brief Loop to calculate the total cost of all bananas.
   * @invariant In each iteration, adds the cost of the i-th banana to the total.
   * The variable `f` is updated to hold the cost of the next banana.
   * 1st loop: ret = k; f = 2k
   * 2nd loop: ret = k + 2k; f = 3k
   * ...
   */
  while (bananas--) {
    ret += f;
    f += eve;
  }
  
  // Check if the soldier has enough money.
  if (ret - init < 0) {
    // If the total cost is less than the initial money, no borrowing is needed.
    cout << 0;
    return 0;
  }
  
  // Otherwise, print the amount to borrow.
  cout << ret - init;
  return 0;
}
