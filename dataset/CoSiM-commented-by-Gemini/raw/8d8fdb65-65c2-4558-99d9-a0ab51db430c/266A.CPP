/**
 * @file 266A.CPP
 * @brief Solution for counting minimum removals to avoid consecutive identical characters.
 * @details This program solves a problem often presented as "Stones on the Table," where the goal
 * is to find the minimum number of stones to remove from a row so that no two adjacent
 * stones have the same color. This is equivalent to counting the number of adjacent,
 * identical characters in a string.
 *
 * @algorithm Linear scan with single-character look-behind.
 * @complexity
 * - Time: O(n), where 'n' is the number of characters (stones), as we process each character once.
 * - Space: O(1), as it uses a constant number of variables for tracking state.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // s: unused string.
  string s;
  // f: current character being read ('first').
  // l: the last character read.
  char f, l;
  // i, j: loop counters (j is unused).
  // x: the count of stones to remove.
  // m, n: n is the number of stones (m is unused).
  long int i, j, x, m, n;

  // This loop structure will process multiple test cases if input is provided that way,
  // but for typical competitive programming platforms, it will run once.
  while (cin >> n) {
    // Initialize the number of stones to remove to zero for this test case.
    x = 0;
    /**
     * @block Character Processing Loop
     * @logic Reads each character one by one and compares it with the previous one.
     */
    for (i = 1; i <= n; i++) {
      cin >> f;
      // Skip the check for the very first stone as it has no predecessor.
      if (i != 1) {
        // If the current stone has the same color as the last one, it needs to be removed.
        if (l == f)
          x++;
      }
      // The current stone now becomes the 'last' stone for the next iteration.
      l = f;
    }
    // Output the total number of stones to remove.
    cout << x << endl;
  }

  return 0;
}
