#include <iostream>

using namespace std;

/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A, "Soldier and Bananas".
 * @details The program calculates the amount of money a soldier needs to borrow
 * to buy a certain number of bananas, where the cost of each subsequent banana
 * increases linearly.
 *
 * Algorithm:
 * The total cost of `w` bananas is the sum of an arithmetic series:
 * `k + 2k + 3k + ... + wk`. Instead of using a direct mathematical formula,
 * this implementation calculates the sum iteratively. A loop runs `w` times,
 * and in each iteration, the cost of the current banana is calculated and
 * added to a running total (`sum`). After the loop, the program checks if the
 * total cost exceeds the soldier's initial money (`n`). If it does, the
 * difference is printed; otherwise, 0 is printed.
 *
 * Time Complexity: O(W), where W is the number of bananas to buy, due to the
 * loop used to calculate the total cost.
 * Space Complexity: O(1).
 */
int main() {
  // k: base cost of a banana.
  // n: initial money the soldier has.
  // w: number of bananas to buy.
  int k, w;
  long long n, sum = 0;
  cin >> k >> n >> w;

  /**
   * @brief Calculate the total cost by iterating and summing up the cost of
   * each banana from 1 to w.
   *
   * Logic: The loop runs `w` times. The variable `j` represents the cost of
   * the i-th banana (k, 2k, 3k, ...).
   */
  for (int i = 0, j = k; i < w; i++, j += k)
    sum += j;
  
  /**
   * @brief Determine the amount to borrow.
   *
   * Pre-condition: `sum` holds the total cost, `n` holds the available money.
   * Logic: If the total cost is more than the available money, the soldier
   *        needs to borrow the difference.
   */
  if (sum > n)
    cout << sum - n;
  else
    cout << 0;

  return 0;
}
