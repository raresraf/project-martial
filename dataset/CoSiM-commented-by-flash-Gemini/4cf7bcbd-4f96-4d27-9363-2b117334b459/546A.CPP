/**
 * @file 546A.CPP
 * @brief Calculates the amount of money needed to borrow to buy a specified number of items (e.g., bananas).
 *        The cost of each subsequent item increases linearly: 1st item costs 'k', 2nd '2*k', ..., w-th 'w*k'.
 *
 * Problem Name: Giga Banana (Codeforces 546A)
 * Algorithm: Arithmetic series sum formula for total cost, followed by a simple subtraction.
 * Time Complexity: O(1) due to direct formula application.
 * Space Complexity: O(1) as it uses a constant number of variables.
 */
/* samikshamodi - Samiksha Modi */
#include <bits/stdc++.h> // Includes most standard C++ libraries, e.g., <iostream>, <string>, <algorithm>, etc.
using namespace std;
typedef long long ll; // Functional Utility: Defines 'll' as a shorthand for 'long long' for larger integer values.

// Macros for common operations, typical in competitive programming.
#define pb push_back   // Functional Utility: Shorthand for vector::push_back.
#define mp make_pair   // Functional Utility: Shorthand for std::make_pair.
#define f first        // Functional Utility: Shorthand for accessing the first element of a pair.
#define s second       // Functional Utility: Shorthand for accessing the second element of a pair.

// Functional Utility: Optimizes I/O operations by decoupling C++ streams from C stdio and untying cin from cout.
#define sync                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
// Functional Utility: Macros for 1D and 2D array input/output and memory initialization.
// These are included for completeness based on typical competitive programming template,
// though not all are used in this specific problem.
#define out1d(arr, size)                                                       \
  {                                                                            \
    for (int i = 0; i < size; i++)                                             \
      cout << arr[i] << " ";                                                   \
  }
#define in2d(arr, row, col)                                                    \
  for (int i = 0; i < row; i++)                                                \
    for (int j = 0; j < col; j++)                                              \
      cin >> arr[i][j];
#define out2d(arr, row, col)                                                   \
  {                                                                            \
    for (int i = 0; i < row; i++) {                                            \
      for (int j = 0; j < col; j++)                                            \
        cout << arr[i][j] << " ";                                              \
      cout << endl;                                                            \
    }                                                                          \
  }
#define mem(arr, val) memset(arr, val, sizeof(arr));

// Constants often used in competitive programming for modulo operations, infinity, and array sizes.
const ll MOD = 1e9 + 7; // Functional Utility: Modulo constant for operations involving large numbers.
const ll INF = 1e18;   // Functional Utility: Represents a large value for infinity.
const ll N = 1e5 + 7;  // Functional Utility: A common maximum array size constant.

/**
 * @brief Main function to calculate the money to be borrowed for purchasing items.
 * @pre Input consists of three integers: 'k' (cost of first item), 'n' (initial money), and 'w' (number of items).
 * @post Prints the amount of money to borrow, or 0 if no borrowing is needed.
 */
int main() {
  sync; // Functional Utility: Calls the I/O optimization macro.

  int k, n, w; // 'k': cost of the first item, 'n': initial money, 'w': number of items to buy.
  ll ans;      // 'ans': Stores the total calculated cost of 'w' items.

  // Functional Utility: Reads the input values for cost, initial money, and number of items.
  cin >> k >> n >> w;

  /**
   * Block Logic: Calculates the total cost of purchasing 'w' items using the arithmetic series sum formula.
   * Invariant: The sum of costs from 1*k to w*k is equivalent to (1+2+...+w)*k.
   *            The sum of 1 to w is w*(w+1)/2.
   */
  ans = ((w * (w + 1)) / 2) * k;

  /**
   * Block Logic: Determines the amount of money to be borrowed.
   * Pre-condition: 'ans' holds the total calculated cost, and 'n' holds the initial money.
   * Invariant: If 'ans' (total cost) is less than 'n' (initial money), no borrowing is needed (result is 0).
   *            Otherwise, the difference ('ans - n') is the amount to borrow.
   */
  if (ans - n < 0)
    cout << 0; // If total cost is less than initial money, print 0 (no borrowing).
  else
    cout << ans - n; // Otherwise, print the amount to borrow.

  // Indicates successful program execution.
  return 0;
}