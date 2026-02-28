/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem using a mathematical formula.
 * @details This program calculates the amount of money a soldier needs to borrow.
 * The cost of the i-th banana is `i*k`. The total cost is calculated using the
 * formula for an arithmetic series sum.
 *
 * @note The variable names in this code are confusingly swapped compared to the
 * problem description. Here, `n` is used for the initial cost `k`, and `k` is
 * used for the initial money `n`.
 *
 * @algorithm Arithmetic Series Summation
 * 1. The total cost is an arithmetic series: `n * 1 + n * 2 + ... + n * w`.
 * 2. This can be factored to `n * (1 + 2 + ... + w)`.
 * 3. The sum of the first `w` integers is `w * (w + 1) / 2`.
 * 4. Therefore, the total cost is `n * (w * (w + 1) / 2)`.
 * 5. The program calculates this cost, compares it to the available money `k`,
 *    and prints the difference if a deficit exists, otherwise 0.
 *
 * @complexity
 * - Time: O(1) per test case, as it's a direct arithmetic calculation.
 * - Space: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  // ans: stores the final amount to borrow.
  // The other variables are confusingly named.
  // n: The cost of the first banana (problem's 'k').
  // k: The soldier's initial money (problem's 'n').
  // w: The number of bananas to buy.
  // m, i, j: temporary variables for calculation.
  ll n, m, i, j, ans, k, w;

  // This loop will run for each test case provided in the input.
  // For this specific problem, there is only one test case.
  while (cin >> n >> k >> w) {
    // `m` calculates the sum of the series 1 + 2 + ... + w.
    m = (w * (w + 1)) / 2;
    
    // `i` calculates the total cost of all bananas.
    // This is `cost_of_first_banana * (sum of 1 to w)`.
    i = n * m;
    
    // Pre-condition: Check if the total cost `i` is greater than the money the soldier has `k`.
    if (i > k) {
      // If there's a deficit, calculate it and print.
      // `abs` is redundant here since `i > k`, but doesn't harm.
      ans = abs(i - k);
      cout << ans << endl;
    } else {
      // If the soldier has enough money, they need to borrow 0.
      cout << "0" << endl;
    }
  }
  return 0; // Indicate successful execution.
}
