/**
 * @file 1366A.CPP
 * @brief Solves the "Shovels and Swords" problem from Codeforces (1366A).
 * @details This program calculates the maximum number of tools (shovels and swords)
 * that can be crafted from a given number of sticks (`a`) and diamonds (`b`).
 * Shovel: 2 sticks, 1 diamond.
 * Sword: 1 stick, 2 diamonds.
 *
 * @section algorithm
 * Algorithm: Constraint Analysis
 * Let `a` be sticks and `b` be diamonds. Each tool, whether a shovel or a sword,
 * requires a total of 3 items (2 of one, 1 of the other).
 * The total number of tools `x` is therefore limited by two primary constraints:
 * 1. The total number of items: `3*x <= a + b`, which implies `x <= (a + b) / 3`.
 * 2. The quantity of the scarcest material: `x <= min(a, b)`. You cannot make more
 *    tools than you have of the least available material.
 * The answer is the minimum of these two upper bounds: `min(min(a, b), (a + b) / 3)`.
 *
 * @section complexity
 * - Time Complexity: O(T) for T test cases, as each case is solved in O(1)
 *   with a direct calculation.
 * - Space Complexity: O(1).
 */
#include <algorithm>
#include <iostream>
using namespace std;

int t;
int a, b;

int main() {
  cin >> t;

  while (t--) {
    int ans = 0;

    cin >> a >> b;

    /**
     * @brief Ensures `a` is the smaller of the two values, so `min(a, b)` can
     * be simplified to just `a`.
     */
    if (a > b) {
      swap(a, b);
    }

    /**
     * @brief Calculates the maximum number of tools.
     * The result is the minimum of the two constraining factors:
     * 1. The count of the scarcest material (`a`).
     * 2. The total number of available item-triplets (`(a + b) / 3`).
     */
    ans = min(a, (a + b) / 3);

    cout << ans << endl;
  }
  return 0; // The return is optional but good practice.
}
