/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem.
 * @details This program calculates the minimum number of stones to remove from a row
 *          such that no two adjacent stones have the same color. This is equivalent
 *          to counting the number of adjacent pairs of stones with the same color.
 *
 * @algorithm The algorithm iterates through the sequence of stones and counts the number of
 *            times a stone is followed by a stone of the same color.
 *
 * @time_complexity O(N), where N is the number of stones, as it requires a single pass.
 * @space_complexity O(N) to store the sequence of stones. For an online version, this
 *                   could be O(1).
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to read input, process, and print the result.
 */
int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  int c = 0;
  /**
   * @brief This loop iterates through the string of stones to find adjacent stones of the same color.
   * @invariant `c` holds the count of adjacent identical stones found in the prefix s[0...i-1].
   */
  for (int i = 0; i < n - 1; i++) {
    // If two adjacent stones have the same color, increment the counter.
    if (s[i] == s[i + 1])
      c++;
  }
  // Output the total number of stones to be removed.
  cout << c;
  return 0;
}
