/**
 * @file 266A.CPP
 * @brief Solution for a problem requiring counting the minimum number of adjacent identical elements.
 * 
 * Algorithm:
 * This program solves the problem of finding the minimum number of stones to remove from a row
 * such that no two adjacent stones are of the same color. It iterates through the sequence of
 * stones and counts every time a stone is identical to the one immediately following it. This
 * count represents the number of stones that need to be removed.
 *
 * Time Complexity: O(N), where N is the number of stones (the length of the string).
 * The algorithm performs a single pass through the string.
 *
 * Space Complexity: O(N), to store the string representing the colors of the stones.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // n: the total number of stones.
  // stones: counter for stones to be removed.
  int n = 0, stones = 0;
  // color: a string representing the colors of the stones in the row.
  string color = "";
  
  // Read the number of stones and their color sequence from standard input.
  cin >> n;
  cin >> color;

  /**
   * @brief Iterates through the sequence of stones to find adjacent duplicates.
   * 
   * Pre-condition: `color` string is populated.
   * Invariant: `stones` holds the count of adjacent identical stones found so far in the scanned part of the string.
   */
  for (int i = 0; i < n - 1; i++) {
    // If the current stone has the same color as the next one,
    // it's a candidate for removal.
    if (color[i] == color[i + 1]) {
      stones++;
    }
  }

  // Output the total number of stones to be removed.
  cout << stones;
  return 0;
}
