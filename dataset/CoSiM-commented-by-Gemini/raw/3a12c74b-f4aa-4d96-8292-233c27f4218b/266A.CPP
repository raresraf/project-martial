/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem (Codeforces 266A).
 *
 * @details The problem asks for the minimum number of stones to remove from a row
 * of colored stones such that no two adjacent stones have the same color.
 *
 * @algorithm The solution involves a single pass through the sequence of stones.
 * The program iterates from the first stone to the second-to-last stone and
 * compares each stone's color (`s[i]`) with the color of the next stone (`s[i+1]`).
 * If the colors are the same, it means one of these two stones must be removed.
 * A counter is incremented for each such adjacent pair. The final value of the
 * counter is the minimum number of stones to remove.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of stones, as it requires a
 *                  single pass through the sequence.
 * Space Complexity: O(N), to store the string of stone colors.
 */
#include <iostream>

using namespace std;

int main() {

  string s;
  int i, n;
  cin >> n >> s;
  int c = 0; // Counter for stones to remove.

  /**
   * @brief Block Logic: Iterates through adjacent pairs of stones.
   * @invariant `c` holds the number of removals needed for the prefix `s[0...i]`.
   */
  for (i = 0; i < n - 1; i++) {
    // If two adjacent stones have the same color, one must be removed.
    if (s[i] == s[i + 1])
      c++;
  }
  cout << c << endl;
}