/********************************************************************************
 *                                                                               *
 *        Author     : Rezoan Shakil Prince * Batch      : 14 * Email      :
 *prince.cuet.77@gmail.com                                  * University :
 *Chittagong University of Engineering and Technology       *
 *                                                                               *
 ********************************************************************************/

#include <bits/stdc++.h>

/**
 * @file 1366A.CPP
 * @brief Solution to a resource allocation problem, "Shovels and Swords".
 *
 * Problem Statement:
 * You have `a` sticks and `b` diamonds. You can craft two types of items:
 * 1. A shovel: requires 1 stick and 2 diamonds.
 * 2. A sword: requires 2 sticks and 1 diamond.
 * Each item is sold for 1 emerald. What is the maximum number of emeralds
 * (i.e., the maximum total number of items) you can get?
 *
 * Mathematical Derivation:
 * Let `x` be the number of shovels and `y` be the number of swords.
 * We are constrained by the available resources:
 * - Sticks:  `x * 1 + y * 2 <= a`
 * - Diamonds: `x * 2 + y * 1 <= b`
 * We want to maximize the total number of items, which is `x + y`.
 *
 * Adding the two inequalities gives:
 * `(x + 2y) + (2x + y) <= a + b`
 * `3x + 3y <= a + b`
 * `3 * (x + y) <= a + b`
 * `x + y <= (a + b) / 3`
 *
 * This means the total number of items `(x + y)` cannot exceed `(a + b) / 3`.
 * Additionally, to make any item, we need at least one stick and one diamond.
 * So, the total number of items is also limited by `a` and `b`.
 * Therefore, the maximum number of items is `min(a, b, (a + b) / 3)`.
 *
 * Time Complexity: O(1) for each test case.
 * Space Complexity: O(1).
 */

using namespace std;

/**
 * @def IOS
 * @brief A macro for optimizing C++ stream I/O operations.
 * It disables synchronization with C-style I/O and unties cin/cout.
 */
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"

typedef long long ll;

/**
 * @brief Solves a single test case for the "Shovels and Swords" problem.
 * @note The `while(cin >> a >> b)` is unconventional for a function that is
 * already called inside a loop for test cases. It's likely intended to just
 * read one pair of `a` and `b` per call.
 */
void solve() {
  ll a, b;
  while (cin >> a >> b)
    // The solution is the minimum of the number of sticks, the number of
    // diamonds, and one-third of their sum. This represents the tightest
    // constraint on the total number of items that can be crafted.
    cout << min({a, b, (a + b) / 3}) << endl;
}

int main() {
  // Apply I/O optimizations.
  IOS;

  // The commented-out code is for redirecting I/O from files,
  // a common practice in competitive programming for local testing.
  // #ifndef prince77
  //         freopen("input.txt", "r", stdin);
  //         freopen("output.txt", "w", stdout);
  // #endif

  int tc;
  // Read the number of test cases.
  cin >> tc;
  // Process each test case.
  while (tc--)
    solve();

  return 0;
}