/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem with an O(1) formula.
 * @author samikshamodi
 * 
 * @details This program calculates the amount of money a soldier needs to borrow.
 * The cost of bananas increases linearly (1k, 2k, 3k, ...). This solution uses a
 * direct mathematical formula to find the total cost, which is more efficient
 * than iterative summation.
 *
 * @algorithm
 * 1. The total cost of `w` bananas is the sum of an arithmetic series:
 *    k * (1 + 2 + 3 + ... + w).
 * 2. This sum is calculated directly using the formula: `total_cost = k * w * (w + 1) / 2`.
 * 3. The amount to borrow is the total cost minus the soldier's initial money `n`.
 *    If the result is negative, it means the soldier has enough money, so 0 is borrowed.
 *
 * @complexity
 * Time Complexity: O(1), as the calculation is done via a direct formula.
 * Space Complexity: O(1).
 */

/* samikshamodi - Samiksha Modi */
#include <bits/stdc++.h>
using namespace std;

// The following are common competitive programming macros for brevity.
// Most are not used in this specific problem.
typedef long long ll;
#define pb push_back
#define mp make_pair
#define f first
#define s second

// Macro for fast I/O
#define sync                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);

// Unused macros for I/O and memory operations.
#define out1d(arr, size)                                                       \
  {                                                                            \
    for (int i = 0; i < size; i++)                                             \
      cout << arr[i] << " ";                                                   \
  }
#define in2d(arr, row, col)
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      cin >> arr[i][j];
#define out2d(arr, row, col)                                                   \
  {                                                                            \
    for (int i = 0; i < row; i++) {                                            \
      for (int j = 0; j < col; j++)
        cout << arr[i][j] << " ";                                              \
      cout << endl;
    }                                                                          \
  }
#define mem(arr, val) memset(arr, val, sizeof(arr));

// Unused constants, likely part of a template.
const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const ll N = 1e5 + 7;

int main() {
  // Apply fast I/O optimization.
  sync;

  ll k, n, w; // k: initial cost, n: initial money, w: number of bananas.
  ll total_cost;
  cin >> k >> n >> w;

  /**
   * @brief Calculate the total cost using the arithmetic series sum formula.
   * This is an O(1) calculation.
   */
  total_cost = ((w * (w + 1)) / 2) * k;

  /**
   * @block
   * @brief Determine the amount to borrow.
   */
  if (total_cost - n < 0)
    // If initial money `n` is more than the cost, no borrowing is needed.
    cout << 0;
  else
    // Otherwise, the amount to borrow is the difference.
    cout << total_cost - n;

  return 0;
}
