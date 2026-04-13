/**
 * @file 1366A.CPP
 * @brief This program solves the "Shovels and Swords" problem.
 * Given `a` units of wood and `b` units of iron, it calculates the maximum
 * number of items (shovels or swords) that can be crafted. Each item requires
 * either (1 wood, 2 iron) or (2 wood, 1 iron).
 * Algorithm: The problem reduces to finding the maximum `k` such that `k` items
 * can be crafted. This is limited by the minimum of `a`, `b`, and `(a+b)/3`.
 * Time Complexity: O(T) due to processing T test cases, with each case being O(1).
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience, includes many standard libraries.
using namespace std;     // Brings all names from the std namespace into the current scope.

// Macros for competitive programming convenience.
#define mod 1000000007                            // A common modulo constant.
#define gcd(a, b) __gcd(a, b)                     // Macro for greatest common divisor.
#define lcm(a, b) (a*(b/gcd(a,b)))                 // Macro for least common multiple.
#define bits(x) __builtin_popcountll(x)           // Counts set bits in a long long.
#define vi vector<int>                            // Alias for vector of integers.
#define vl vector<ll>                             // Alias for vector of long long integers.
#define all(v) v.begin(), v.end()                 // Returns iterators for the entire range of a container.
#define UNIQUE(X) (X).erase(unique(all(X)), (X).end()) // Removes duplicate elements from a sorted container.

typedef long long int ll; // Alias for long long integer.

/**
 * @brief Solves a single test case for the "Shovels and Swords" problem.
 * Reads the amounts of wood and iron, and prints the maximum number of items craftable.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
void solve() {
  ll a, b; // 'a' represents units of wood, 'b' represents units of iron.
  // Reads the units of wood and iron for the current test case.
  cin >> a >> b;
  // Prints the maximum number of items craftable. This is the minimum of:
  // 1. The number of wood units 'a'.
  // 2. The number of iron units 'b'.
  // 3. The total resources divided by 3 (since each item effectively consumes 3 total resource units).
  cout << min(a, min(b, (a + b) / 3)) << "\n";
}

int main() {
  // Disables synchronization between C-style I/O and C++ streams, and unties cin from cout.
  // This optimizes input/output operations for competitive programming.
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  ll t; // Stores the number of test cases.
  // Reads the number of test cases.
  cin >> t;
  // cout<<t<<"Hi test complete\n"; // Commented-out debugging line.
  /**
   * Block Logic: Processes 't' test cases.
   * Each iteration calls the `solve()` function to handle one test case.
   * Invariant: `t` decrements until all test cases are processed.
   */
  while (t--) {
    solve(); // Calls the function to solve the current test case.
  }

  // Standard program exit.
  return 0;
}