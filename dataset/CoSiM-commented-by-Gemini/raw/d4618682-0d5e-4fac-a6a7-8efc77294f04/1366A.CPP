/**
 * @file 1366A.CPP
 * @brief Solves the "Shovels and Swords" problem (Codeforces 1366A).
 * @author prajwal1721
 * 
 * @details This program calculates the maximum number of tools (shovels or swords)
 * that can be crafted from 'a' sticks and 'b' diamonds. This solution uses a
 * common competitive programming template with macros and a concise solve function.
 *
 * @section algorithm
 * Algorithm: Constraint Analysis
 * Each tool (shovel or sword) requires a total of 3 items. The total number of
 * tools, `x`, is therefore limited by three upper bounds:
 * 1. `x <= a` (cannot make more tools than available sticks)
 * 2. `x <= b` (cannot make more tools than available diamonds)
 * 3. `3*x <= a + b` (cannot make more tools than the total number of item-triplets)
 * The solution is the minimum of these three bounds, which is concisely
 * expressed as `min(a, min(b, (a + b) / 3))`.
 *
 * @section complexity
 * - Time Complexity: O(T) for T test cases. Each test case is solved in O(1)
 *   via a direct mathematical calculation.
 * - Space Complexity: O(1).
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Common macros for competitive programming to shorten code.
 * mod: A prime modulus for modular arithmetic.
 * gcd/lcm: Greatest common divisor and least common multiple.
 * bits: Counts set bits in a long long.
 * vi/vl: Typedefs for vector<int> and vector<ll>.
 * all(v): Returns an iterator pair for a container's beginning and end.
 * UNIQUE(X): Removes duplicate elements from a sorted container.
 */
#define mod 1000000007
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) (a*(b/gcd(a,b)))
#define bits(x) __builtin_popcountll(x)
#define vi vector<int>
#define vl vector<ll>
#define all(v) v.begin(), v.end()
#define UNIQUE(X) (X).erase(unique(all(X)), (X).end())

typedef long long int ll;

/**
 * @brief Processes a single test case.
 */
void solve() {
  ll a, b;
  cin >> a >> b;
  
  // Directly applies the formula derived from the problem's constraints.
  cout << min(a, min(b, (a + b) / 3)) << "\n";
}

int main() {
  /**
   * @brief Optimizes C++ standard streams for faster I/O in competitive programming.
   */
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  
  ll t;
  cin >> t;
  // Loop to handle multiple test cases.
  while (t--) {
    solve();
  }

  return 0;
}