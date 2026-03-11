/**
 * @file 231A.CPP
 * @brief Solution to Codeforces Problem 231A: "Team".
 *        The problem involves determining how many problems a team of three friends
 *        will implement in a programming contest. They will implement a problem
 *        if at least two of them are sure about the solution.
 *
 * Algorithm: Iterative counting. For each problem, read the certainty levels of the three friends.
 *            Sum these levels, and if the sum is 2 or more, increment a counter for implemented problems.
 * Time Complexity: O(N) where N is the number of problems, as each problem requires a fixed number
 *                  of input operations and a simple sum/comparison.
 * Space Complexity: O(1) as only a few variables are used to store problem-specific data.
 */
// .... .... .... !

#include <fstream>
#include <iomanip>
#include <iostream>

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>

#include <cstdio>
#include <cstring>

using namespace std;

/**
 * @brief Macro for a common for-loop pattern, iterating from 0 up to n (exclusive).
 * Functional Utility: Simplifies writing loops for 0-indexed arrays or collections.
 * @param i: Loop variable.
 * @param n: Upper bound (exclusive) of the loop.
 */
#define rep(i, n) for (int i = 0, _n = (int)(n); i < _n; i++)
/**
 * @brief Macro for a flexible for-loop pattern, iterating from x up to n (exclusive).
 * Functional Utility: Provides a generalized loop for arbitrary start and end points.
 * @param i: Loop variable.
 * @param x: Lower bound (inclusive) of the loop.
 * @param n: Upper bound (exclusive) of the loop.
 */
#define fer(i, x, n) for (int i = (int)(x), _n = (int)(n); i < _n; i++)
/**
 * @brief Macro for a reverse for-loop pattern, iterating from n (exclusive) down to x (exclusive).
 * Functional Utility: Simplifies reverse iteration.
 * @param i: Loop variable.
 * @param n: Upper bound (exclusive) of the loop.
 * @param x: Lower bound (exclusive) of the loop.
 */
#define rof(i, n, x) for (int i = (int)(n), _x = (int)(x); i-- > _x;)
/**
 * @brief Macro for iterating through elements of a container (e.g., vector, map).
 * Functional Utility: Provides a convenient way to iterate over generic containers.
 * @param i: Iterator variable.
 * @param x: The container to iterate over.
 */
#define fch(i, x) for (__typeof(x.begin()) i = x.begin(); i != x.end(); i++)
/**
 * @brief Macro for getting the size of a container.
 * Functional Utility: Casts container size to an int for consistency.
 * @param x: The container.
 */
#define sz(x) (int((x).size()))
/**
 * @brief Macro for accessing the first element of a pair or map entry.
 * Functional Utility: Shorthand for `pair.first`.
 */
#define X first
/**
 * @brief Macro for accessing the second element of a pair or map entry.
 * Functional Utility: Shorthand for `pair.second`.
 */
#define Y second

/**
 * @brief Type definition for long long integer.
 * Functional Utility: Provides a shorter alias for `long long`.
 */
typedef long long ll;
/**
 * @brief Type definition for a pair of integers.
 * Functional Utility: Provides a shorter alias for `pair<int, int>`.
 */
typedef pair<int, int> pii;

////////////////////////////////////////////////////////////////////////////////

int main() {
  /**
   * Functional Utility: Decouples C++ standard streams from C standard streams.
   * This can speed up I/O operations, which is beneficial in competitive programming.
   */
  ios::sync_with_stdio(false);
  int n;
  cin >> n; // Read the number of problems.
  int r = 0; // Initialize counter for problems to be implemented.
  /**
   * Block Logic: Iterates 'n' times, once for each problem.
   * Pre-condition: `n` represents the total number of problems.
   * Invariant: In each iteration, `x`, `y`, `z` represent the certainty of three friends
   *            about the current problem's solution (1 if sure, 0 if not).
   */
  rep(i, n) {
    int s, x, y, z;
    // Input: Read certainty levels of three friends for the current problem.
    cin >> x >> y >> z;
    s = x + y + z; // Calculate the total number of friends sure about the solution.
    /**
     * Conditional Logic: Check if at least two friends are sure about the solution.
     * If `s` (total sure friends) is 2 or more, the team will implement this problem.
     */
    if (s >= 2)
      r++; // Increment the counter for implemented problems.
  }
  // Output: Print the total number of problems the team will implement.
  cout << r << endl;
  /**
   * Functional Utility: This block reads an extra integer from input, likely a common practice
   *                     in some competitive programming platforms to consume trailing input
   *                     or to keep the console open when debugging locally.
   *                     It does not affect the problem's core logic.
   */
  {
    int _;
    cin >> _;
  }
  return 0;
}
