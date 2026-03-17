/**
 * @file 266A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 266A "Stones on the Table".
 *
 * @details
 * This program calculates the minimum number of stones to remove from a row
 * to ensure that no two adjacent stones have the same color. The colors are
 * represented by characters.
 *
 * @algorithm
 * The algorithm is a single linear pass through the sequence of stones. It
 * maintains the color of the previous stone and compares it with the current
 * stone. If the colors are the same, a counter is incremented, signifying that
 * the current stone must be removed. If they are different, the 'previous'
 * color is updated to the current one.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of stones, because it iterates
 *   through the input sequence exactly once.
 * - Space Complexity: O(1), as it only uses a few variables to store state,
 *   regardless of the input size.
 */

#include <cctype>
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  // The line below is commented out but is often used in competitive programming
  // to redirect input from a file for local testing.
  // freopen("input.txt", "r", stdin);
  
  int n; // Total number of stones.
  cin >> n;
  
  char x, y; // 'x' holds the color of the previous stone, 'y' the current.
  int cnt = 0; // Counter for stones to be removed.

  // Pre-condition: Read the first stone to initialize the 'previous stone' color.
  cin >> x;
  
  /**
   * @brief Linear scan through the remaining stones.
   * This loop reads each subsequent stone into 'y' and compares it with 'x'.
   * The loop structure `while (cin >> y)` effectively iterates n-1 times.
   */
  while (cin >> y) {
    /**
     * @brief Core logic for counting removals.
     * If the current stone 'y' has the same color as the previous one 'x',
     * it's a consecutive pair, so the removal counter is incremented.
     * Otherwise, the stones are different, so we update the 'previous'
     * color to the current one for the next comparison.
     */
    if (y != x)
      x = y;
    else
      cnt++;
  }
  
  // Output the total number of stones that need to be removed.
  cout << cnt;
  return 0;
}
