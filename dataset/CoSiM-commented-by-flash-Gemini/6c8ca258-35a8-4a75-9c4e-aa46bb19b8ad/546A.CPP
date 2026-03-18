/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem: "Soldier and Bananas".
 * Calculates the amount of money a soldier needs to borrow to buy 'w' bananas.
 * The first banana costs 'k', the second '2k', and so on.
 * Algorithm: Iterative summation for total cost, then comparison for deficit.
 * Time Complexity: O(W) where W is the number of bananas.
 * Space Complexity: O(1)
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to solve the Soldier and Bananas problem.
 * Reads the cost of the first banana (k), initial money (n), and number of bananas (w).
 * Calculates the total cost of bananas and outputs the amount of money to borrow.
 */
int main() {
  int k, w;
  long long n, sum = 0;
  cin >> k >> n >> w;

  /**
   * Block Logic: Calculates the total cost of 'w' bananas.
   * The cost of the i-th banana is 'i * k'.
   * Pre-condition: 'k', 'n', and 'w' have been read. 'sum' is initialized to 0.
   * Invariant: 'sum' accumulates the cost of bananas from 1 to 'w'.
   */
  for (int i = 0, j = k; i < w; i++, j += k)
    sum += j;
  /**
   * Block Logic: Determines if the soldier needs to borrow money.
   * If the total cost ('sum') is greater than initial money ('n'),
   * calculates the deficit; otherwise, no money needs to be borrowed.
   */
  if (sum > n)
    cout << sum - n;
  else
    cout << 0;

  return 0;
}
