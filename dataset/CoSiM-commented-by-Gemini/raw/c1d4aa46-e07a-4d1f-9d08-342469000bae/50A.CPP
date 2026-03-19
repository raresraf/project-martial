/**
 * @file 50A.CPP
 * @brief Solves a tiling problem using a recursive approach, with a flawed memoization attempt.
 * @details
 * This program is likely intended to solve the "Domino Piling" problem (Codeforces 50A),
 * which asks for the maximum number of 2x1 dominoes on an M x N board.
 * The standard solution is a simple formula: (M * N) / 2.
 *
 * This implementation uses a complex recursive algorithm. It attempts to use memoization
 * with the `ans` table, but it fails to store the computed results, leading to
 * exponential time complexity instead of the intended polynomial time.
 * The recursive logic itself is also non-standard and may not correctly solve the problem.
 *
 * Algorithm: Flawed Dynamic Programming (Memoization check without store).
 * Time Complexity: Exponential due to the lack of result storage in the memoization table.
 * Space Complexity: O(M * N) for the memoization table.
 */
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

typedef unsigned long long ULL;
typedef long long LL;

// Macro for a for-loop from 0 to n-1.
#define REP(i, n) FOR(i, 0, n)
// Macro for a standard for-loop.
#define FOR(i, a, b) for (int i = a; i < b; i++)
// Macro for a reverse for-loop.
#define ROF(i, a, b) for (int i = a; i > b; i--)
/**
 * @brief Macro to read an integer from standard input using scanf.
 * Provides a concise way to get integer input.
 */
#define GI                                                                     \
  ({	
    int t;
    scanf("%d", &t);
    t;
  })
/**
 * @brief Macro to read a long long from standard input using cin.
 */
#define GL                                                                     \
  ({
    LL t;
    cin >> t;
    t;
  })
/**
 * @brief Macro to read a double from standard input using scanf.
 */
#define GD                                                                     \
  ({
    double t;
    scanf("%lf", &t);
    t;
  })

// Shortcut for the push_back vector operation.
#define pb push_back
// Shortcut for the make_pair operation.
#define mp make_pair
#define MOD 1000000007
#define INF (int)1e8
#define EPS 1e-9
// Macro to iterate through a container's elements.
#define TR(a, it)                                                              
  for (typeof((a).begin()) it = (a).begin(); it != (a).end(); ++it)

// A 17x17 memoization table to store results of subproblems.
int ans[17][17];

/**
 * @brief Recursively computes a value based on grid dimensions m and n.
 * @param m The first dimension of the grid.
 * @param n The second dimension of the grid.
 * @return The computed value for the given dimensions.
 */
int solve(int m, int n) {
  // Normalize dimensions so that m is always the larger one.
  int sum = m + n;
  m = max(m, n);
  n = sum - m;

  // Base case: If the grid is too small for the recursive step's logic, return 0.
  if (m < 2 || n < 1)
    return 0;

  // Pre-condition: Check if the solution for (m, n) is already memoized.
  // Invariant: If ans[m][n] is not -1, it holds a valid pre-computed result.
  if (ans[m][n] != -1)
    return ans[m][n];

  int t1 = 0, t2 = 0;

  // Block Logic: Recursive decomposition. The logic seems to break down the
  // problem into smaller, overlapping subproblems, but the state transitions
  // are non-obvious for a standard tiling problem.
  if (m >= 2 && n >= 1)
    t1 = solve(m, n - 1) + solve(m - 2, 1);
  if (m >= 1 && n >= 2)
    t2 = solve(m, n - 2) + solve(m - 1, 2);

  // Combine results from subproblems.
  int res = 1 + max(t1, t2);

  // CRITICAL FLAW: The result 'res' is not stored in the memoization table 'ans'.
  // This means the function recomputes the same states repeatedly,
  // leading to exponential time complexity.
  return res;
}

int main() {
  // Block Logic: Initialize the memoization table.
  // Pre-condition: ans table is uninitialized.
  // Invariant: All cells of ans are set to -1 to indicate they are not yet computed.
  FOR(i, 0, 17)
  FOR(j, 0, 17)
  ans[i][j] = -1;

  // Read the grid dimensions from standard input.
  int m = GI;
  int n = GI;

  // Call the recursive solver and print the result.
  cout << solve(m, n);

  return 0;
}
