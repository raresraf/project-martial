/**
 * @file 546A.CPP
 * @brief Calculates the amount of money a soldier needs to borrow to buy bananas.
 *
 * This program solves the "Soldier and Bananas" problem from Codeforces.
 * It calculates the total cost of buying a specific number of bananas where each
 * subsequent banana costs more than the previous one in an arithmetic progression.
 * It then determines how much money the soldier needs to borrow, if any.
 *
 * @algorithm Arithmetic Series Summation
 * @complexity
 * - Time: O(W), where W is the number of bananas to buy, due to the loop that
 *   calculates the total price. A more optimized O(1) solution could use the
 *   formula for the sum of an arithmetic series: cost = k * W * (W + 1) / 2.
 * - Space: O(1), as it uses a fixed number of variables.
 */
#include <iostream>
#include <cmath> // For abs()

using namespace std;

/**
 * @note The following line is syntactically incorrect for its likely intent.
 * It should be `typedef long long ll;` or `using ll = long long;`.
 * As `ll` is unused, it does not cause a compilation error.
 */
#define long long ll;

// Global variables for the problem parameters.
int k, n, w; // k: cost of first banana, n: initial money, w: bananas to buy.

/**
 * @brief Core logic to calculate and print the amount to borrow.
 */
void solve() {
  int totalprice = 0;

  /**
   * @brief Block Logic: Calculates the total cost of `w` bananas.
   * Invariant: `totalprice` accumulates the cost. The cost of the i-th banana is `i * k`.
   */
  for (int i = 1; i <= w; i++) {
    totalprice += i * k;
  }

  // Calculate the difference between money owned and money needed.
  int solution = n - totalprice;

  /**
   * @brief Block Logic: Determines the amount to borrow.
   * If `solution` is positive, the soldier has enough money, so nothing is borrowed.
   * If `solution` is negative, its absolute value is the amount to borrow.
   */
  if (solution > 0)
    solution = 0; // The soldier doesn't need to borrow any money.

  // An alternative to the if/abs logic is: cout << max(0, totalprice - n);
  cout << abs(solution);
}

int main() {
  cin >> k >> n >> w;
  solve();
}
