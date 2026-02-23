/**
 * @file 266A.CPP
 * @brief Solution to Codeforces problem 266A - Stones on the Table.
 *
 * @algorithm
 * The problem asks for the minimum number of stones to remove from a row
 * so that no two adjacent stones have the same color. This is equivalent to
 * counting the number of adjacent pairs of stones with the same color.
 *
 * The implemented algorithm is as follows:
 * 1. Read the number of stones `n` and the string `s` representing their colors.
 * 2. Initialize a counter `cnt` to 0. This will store the number of stones to remove.
 * 3. A character `a` is used to keep track of the color of the previous stone. It's initialized with the color of the first stone `s[0]`.
 * 4. The code iterates through the string from the second stone (index 1).
 * 5. In each iteration, it compares the color of the current stone `s[i]` with the color
 *    of the previously tracked stone `a`.
 * 6. If they are the same, it means a stone needs to be removed, so `cnt` is incremented.
 * 7. The tracking character `a` is then updated to the color of the current stone `s[i]` for the next iteration.
 * 8. After the loop, `cnt` holds the total count, which is the minimum number of removals.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of stones (the length of the string),
 *   due to a single pass through the string.
 * - Space Complexity: O(N) to store the string of stone colors.
 */
#include <iostream>
using namespace std;
int main() {
  // n: the total number of stones on the table.
  int n;
  cin >> n;
  // s: a string representing the colors of the stones.
  string s;
  cin >> s;
  // cnt: counter for the number of stones to be removed.
  int cnt = 0;
  // a: stores the color of the stone from the previous position for comparison.
  char a = s[0];
  /**
   * Block Logic: Iterate through the stones starting from the second stone
   * to find adjacent stones of the same color.
   * Pre-condition: `n` is the length of `s`, `a` is `s[0]`, `cnt` is 0.
   * Invariant: `cnt` stores the number of removals needed for the first `i` stones,
   * and `a` stores the color of the (i-1)-th stone.
   */
  for (int i = 1; i < n; i++) {
    // Check if the current stone's color is the same as the previous one.
    if (a == s[i]) {
      // If they are the same, one must be removed.
      cnt++;
    }
    // Update the previous stone's color for the next iteration.
    a = s[i];
  }
  // Output the total minimum number of stones to remove.
  cout << cnt;
}