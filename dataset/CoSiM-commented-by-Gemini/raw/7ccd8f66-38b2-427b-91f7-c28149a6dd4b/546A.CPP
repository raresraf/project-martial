/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A, "Soldier and Bananas".
 * @details The program calculates the amount of money a soldier needs to borrow
 * to buy `w` bananas, where the cost increases for each banana.
 *
 * Algorithm:
 * 1. Read the initial cost of the first banana (`k`), the soldier's starting
 *    money (`n`), and the number of bananas he wants (`w`).
 * 2. Calculate the total cost by summing the prices in a loop. The cost of the
 *    i-th banana is `i * k`. The loop runs from i=1 to w.
 * 3. Calculate the difference between the total cost and the money the soldier
 *    has. This is the potential amount to borrow.
 * 4. If the difference is negative (meaning the soldier has enough money), the
 *    amount to borrow is 0.
 * 5. Print the final result.
 *
 * Time Complexity: O(W), where W is the number of bananas, due to the loop.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;

  int total = 0;
  /**
   * @brief This loop calculates the total cost of `w` bananas, where the
   * cost for the i-th banana is `i * k`.
   */
  for (int i = 1; i <= w; i++)
    total += i * k;

  // Calculate the amount to borrow.
  int result = total - n;

  /**
   * @brief If the result is negative, it means the soldier has enough money,
   * so the amount to borrow is 0.
   */
  if (result < 0)
    result = 0;
  cout << result << endl;

  return 0;
}