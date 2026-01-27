/**
 * @file 1366A.CPP
 * @brief Solution for Codeforces problem 1366A "Shovels and Swords".
 *
 * This program calculates the maximum number of items (e.g., shovels/swords)
 * that can be crafted given two types of resources, 'a' and 'b'.
 * The crafting rule is to combine one of type 'a' and two of type 'b', or
 * two of type 'a' and one of type 'b'.
 *
 * Algorithm: The maximum number of items is limited by two conditions:
 * 1. The total number of pairs (a+b)/3.
 * 2. The minimum of 'a' and 'b', as each item requires at least one of each.
 * The final answer is the minimum of these two limiting factors.
 *
 * Time Complexity: O(1) per test case.
 * Space Complexity: O(1).
 */
// Author Tejas K Mukherjee
#include <bits/stdc++.h>
using namespace std;
#define ll long long // Functional Utility: Macro for `long long` data type, commonly used in competitive programming for larger integer values.
#define FOR(a, b) for (int i = a; i < b; i++) // Functional Utility: Macro for a common for-loop pattern (inclusive start, exclusive end).
#define FORE(a, b) for (int i = a; i <= b; i++) // Functional Utility: Macro for a common for-loop pattern (inclusive start, inclusive end).
int main() {
  int t;
  // Block Logic: Reads the number of test cases.
  cin >> t;
  // Block Logic: Iterates through each test case.
  while (t--) {
    ll a, b;
    // Functional Utility: Reads two long long integers representing resource counts.
    cin >> a >> b;
    /**
     * Block Logic: Calculates the maximum number of items that can be crafted.
     * The logic is `min((a + b) / 3, min(a, b))`.
     * Explanation:
     * 1. `(a + b) / 3`: Represents the maximum number of items if resources were only limited by the total sum (each item consumes 3 units in total).
     * 2. `min(a, b)`: Represents the maximum number of items if resources were limited by the smaller of the two resource types (each item requires at least one of each type).
     * The actual maximum is the bottleneck, i.e., the minimum of these two limiting factors.
     */
    cout << min((a + b) / 3, min(a, b)) << '\n';
  }
}