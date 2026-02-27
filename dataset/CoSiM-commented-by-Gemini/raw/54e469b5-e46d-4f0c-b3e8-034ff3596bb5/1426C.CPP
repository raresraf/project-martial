/**
 * @file 1426C.CPP
 * @brief Solves the "Increase and Copy" problem from Codeforces.
 * @details The problem asks for the minimum number of moves to obtain a value
 * of at least 'n', starting from 1. The available moves are:
 * 1. Increment the current number by 1.
 * 2. Copy the current number (effectively doubling it, tripling, etc., by adding
 *    the current number to itself).
 *
 * @algorithm
 * This is an optimization problem. Let's assume we decide to use the 'copy'
 * operation when our current number is `i`.
 * - To reach the value `i` from 1 requires `i - 1` increment moves.
 * - Once we have `i`, we need to reach `n`. This will take `ceil(n / i) - 1`
 *   copy moves. (We need a total of `ceil(n/i)` copies of `i`, and we already have one).
 * - So, for a chosen copy value `i`, the total moves are `(i - 1) + (ceil(n / i) - 1)`.
 *
 * To find the minimum total moves, we need to find an `i` that minimizes this
 * expression. The expression `i + n/i` is minimized when `i` is close to `sqrt(n)`.
 * Therefore, it is sufficient to check values of `i` up to `sqrt(n)`.
 *
 * The code iterates through `i` from 1 to `sqrt(n)`, calculates the total moves
 * for each `i`, and keeps track of the minimum.
 *
 * @complexity
 * Time Complexity: O(sqrt(n)) for each test case, due to the loop.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

// Macro for fast I/O in competitive programming.
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);

typedef long long int ll;

int main() {
  optimizar_io 
  int tc, n;

  cin >> tc; // Number of test cases.
  while (tc--) {
    cin >> n;
    int ans = numeric_limits<int>::max(); // Initialize min steps to a large value.

    /**
     * @block
     * @brief Iterate through possible values to use for the 'copy' operation.
     * We only need to check up to sqrt(n) because the function `i + n/i` is
     * minimized when `i` is close to `sqrt(n)`.
     */
    for (int i = 1; i * i <= n; i++) {
      // Moves to reach 'i' is `i - 1`.
      int steps = i - 1;
      
      // Moves to reach 'n' from 'i' using copies is `ceil(n/i) - 1`.
      // The code calculates this as `(n/i) - 1` and adds 1 if there's a remainder.
      int rem = n % i;
      steps += (n / i) - 1;
      if (rem)
        steps++;
      
      // Update the minimum steps found so far.
      ans = min(ans, steps);
    }
    cout << ans << "\n";
  }
  return 0;
}
