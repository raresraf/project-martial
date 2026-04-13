/**
 * @file 50A.CPP
 * @brief Dynamic Programming solution for a combinatorial problem with two dimensions.
 *
 * This program solves a recursive problem characterized by two dimensions, 'm' and 'n'.
 * It aims to find the maximum count of placements or operations that can be performed
 * by consuming resources from these two dimensions in specific patterns. The solution
 * uses dynamic programming with memoization to avoid redundant computations.
 *
 * This problem is NOT the standard "Domino Piling" problem despite the filename,
 * but rather a more complex tiling or resource allocation variant.
 *
 * Algorithm: Recursive Dynamic Programming with Memoization.
 * Time Complexity: O(M*N), where M and N are the maximum possible values for the dimensions
 *                  (17 in this case), due to the memoization table of size 17x17.
 * Space Complexity: O(M*N) for the memoization table `ans[17][17]` and the recursion stack.
 */
#include <algorithm> // Required for std::max.
#include <cctype>    // For character classification (not used in main logic).
#include <climits>   // For integer limits.
#include <cmath>     // For mathematical functions.
#include <cstdio>    // For C-style input/output (scanf).
#include <cstdlib>   // For general utilities.
#include <cstring>   // For C-style string manipulation.
#include <iostream>  // For C++ style input/output (cin, cout).
#include <list>      // For std::list (not used in main logic).
#include <map>       // For std::map (not used in main logic).
#include <queue>     // For std::queue (not used in main logic).
#include <set>       // For std::set (not used in main logic).
#include <stack>     // For std::stack (not used in main logic).
#include <string>    // For std::string (not used in main logic).
#include <vector>    // For std::vector (not used in main logic).

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

// Type definitions for common data types.
typedef unsigned long long ULL; ///< @brief Alias for unsigned long long integer.
typedef long long LL;           ///< @brief Alias for long long integer.

// Macros for common looping patterns, input/output, and utilities.
#define REP(i, n) FOR(i, 0, n) ///< @brief Loop from 0 (inclusive) to 'n' (exclusive).
#define FOR(i, a, b) for (int i = a; i < b; i++) ///< @brief Loop from 'a' (inclusive) to 'b' (exclusive).
#define ROF(i, a, b) for (int i = a; i > b; i--) ///< @brief Loop from 'a' (inclusive) down to 'b'+1 (exclusive).
// Macro for fast integer input using scanf.
#define GI                                                                     
  ({                                                                           
    int t;                                                                     
    scanf("%d", &t);                                                           
    t;                                                                         
  })
// Macro for fast long long input using cin.
#define GL                                                                     
  ({                                                                           
    LL t;                                                                      
    cin >> t;                                                                  
    t;                                                                         
  })
// Macro for fast double input using scanf.
#define GD                                                                     
  ({                                                                           
    double t;                                                                  
    scanf("%lf", &t);                                                          
    t;                                                                         
  })
#define pb push_back ///< @brief Appends an element to a std::vector.
#define mp make_pair ///< @brief Creates a std::pair.
#define MOD 1000000007 ///< @brief Constant for modulo operation.
#define INF (int)1e8 ///< @brief Integer infinity constant.
#define EPS 1e-9     ///< @brief Epsilon for floating-point comparisons.
// Macro for iterating through containers (type-safe for C++03, older style).
#define TR(a, it)                                                              
  for (typeof((a).begin()) it = (a).begin(); it != (a).end(); ++it)

// Global 2D array used for memoization in the `solve` function.
// Initialized to -1 to indicate uncomputed states.
int ans[17][17];

/**
 * @brief Recursive dynamic programming function to solve the combinatorial problem.
 *
 * This function calculates the maximum number of placements or operations possible
 * given two dimensions `m` and `n`. It uses memoization (stored in the global `ans` array)
 * to store results of subproblems and avoid redundant calculations.
 *
 * @param m The first dimension's current value.
 * @param n The second dimension's current value.
 * @return The maximum number of placements/operations possible from the state (m, n).
 */
int solve(int m, int n) {
  // Ensure m is always the larger dimension for canonical state representation,
  // potentially for memoization table symmetry or simplified logic.
  int sum = m + n;
  m = max(m, n);
  n = sum - m; // Now n holds the smaller of the original m, n.

  // Base case: If resources are insufficient, no more placements can be made.
  if (m < 2 || n < 1)
    return 0;
  // Memoization: If the result for this state (m, n) has already been computed, return it.
  if (ans[m][n] != -1)
    return ans[m][n];

  int t1 = 0, t2 = 0; // Variables to store results of two possible choices/branches.

  // Choice 1: Consumes resources in a specific pattern.
  // This likely represents a placement that reduces 'n' by 1 and 'm' by 2,
  // combined with a placement that effectively replaces the consumed 'm' with '1'.
  if (m >= 2 && n >= 1)
    t1 = solve(m, n - 1) + solve(m - 2, 1);
  // Choice 2: Consumes resources in another specific pattern.
  // This likely represents a placement that reduces 'n' by 2 and 'm' by 1,
  // combined with a placement that effectively replaces the consumed 'n' with '2'.
  if (m >= 1 && n >= 2)
    t2 = solve(m, n - 2) + solve(m - 1, 2);

  // The result for the current state is 1 (for the current placement) plus the
  // maximum result from the subsequent recursive choices.
  int res = 1 + max(t1, t2);
  // Store the computed result in the memoization table before returning.
  return ans[m][n] = res;
}

/**
 * @brief Main function: entry point of the program.
 * Initializes the memoization table, reads the initial dimensions M and N,
 * and then calls the `solve` function to compute and print the maximum
 * number of placements.
 */
int main() {
  // Initializes the memoization table `ans` with -1, marking all states as uncomputed.
  FOR(i, 0, 17)
  FOR(j, 0, 17)
  ans[i][j] = -1;

  int m = GI; // Reads the initial value for the first dimension M using the GI macro.
  int n = GI; // Reads the initial value for the second dimension N using the GI macro.
  cout << solve(m, n); // Calls the solve function and prints its result.

  return 0; // Indicates successful program execution.
}