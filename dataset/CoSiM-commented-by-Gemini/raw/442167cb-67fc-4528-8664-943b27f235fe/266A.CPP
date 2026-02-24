/**
 * @file raw/442167cb-67fc-4528-8664-943b27f235fe/266A.CPP
 * @brief Solution to Codeforces problem 266A - Stones on the Table.
 *
 * @algorithm
 * The problem asks for the minimum number of stones to remove from a table
 * such that no two adjacent stones have the same color. This is equivalent to
 * counting the number of pairs of adjacent stones that have the same color.
 *
 * The algorithm implemented is as follows:
 * 1. Read the number of stones, `n`, and the sequence of stone colors as a string `s`.
 * 2. Initialize a counter `cnt` to 0, which will store the number of stones to remove.
 * 3. A character variable `a` is used to keep track of the color of the previously
 *    inspected stone. It is initialized with the color of the first stone, `s[0]`.
 * 4. Iterate through the string from the second character (`i = 1`) to the end.
 * 5. In each iteration, compare the color of the current stone `s[i]` with the
 *    color stored in `a`.
 * 6. If they are the same, it means we have two adjacent stones of the same color,
 *    so one must be removed. The counter `cnt` is incremented.
 * 7. The variable `a` is updated to the color of the current stone `s[i]` to be
 *    used in the next iteration's comparison.
 * 8. After the loop, `cnt` holds the total count of minimum removals needed.
 * 9. Print the final count.
 *
 * @complexity
 * - Time Complexity: O(n), where n is the number of stones, due to the single
 *   pass through the string.
 * - Space Complexity: O(n) to store the input string `s`. The auxiliary space
 *   complexity is O(1).
 */
#include <iostream>
using namespace std;
int main() {
  // n: the number of stones on the table.
  int n;
  cin >> n;
  // s: a string representing the colors of the stones.
  string s;
  cin >> s;

  // cnt: counts the number of stones to remove.
  int cnt = 0;
  // a: holds the color of the previous stone for comparison.
  char a = s[0];

  /**
   * @brief This loop iterates through the stones to count adjacent duplicates.
   *
   * Pre-condition: `a` holds the color of the stone at index `i-1`.
   * Invariant: `cnt` stores the number of removals needed for the first `i` stones.
   */
  for (int i = 1; i < n; i++) {
    // If the current stone's color matches the previous one...
    if (a == s[i]) {
      // ...we need to remove one of them.
      cnt++;
    }
    // Update the previous stone's color for the next iteration.
    a = s[i];
  }

  // Output the total number of stones to be removed.
  cout << cnt;
}
