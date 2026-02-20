/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove to avoid adjacent same-colored stones.
 *
 * This program solves the "Stones on the Table" problem. It calculates the minimum
 * number of stones to take from a row so that no two adjacent stones have the
 * same color. This is equivalent to counting the number of pairs of adjacent,
 * identical characters in a string.
 *
 * Algorithm:
 * 1. Read the number of stones (n) and the string representing the colors (s).
 * 2. Initialize a counter for removals to 0.
 * 3. Iterate through the string from the second character.
 * 4. Compare each character with the previous one.
 * 5. If they are the same, increment the counter.
 * 6. The final count represents the minimum number of removals.
 *
 * Time Complexity: O(N), where N is the number of stones (the length of the string),
 * as it requires a single pass through the string.
 * Space Complexity: O(N) to store the input string.
 */
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int n;
  string s;
  // Read the number of stones and their colors.
  cin >> n;
  cin >> s;

  char a; // Stores the color of the previous stone.
  int count = 0; // Counts the number of stones to remove.

  // Pre-condition: The string must not be empty. Initialize with the first stone's color.
  a = s[0];

  /**
   * Block Logic: Iterate through the stones to find adjacent pairs.
   * This loop starts from the second stone (index 1) and compares each stone
   * with the color of the stone preceding it.
   */
  for (int i = 1; i < s.length(); i++) {
    // Invariant: 'a' holds the color of the stone at index i-1.
    if (a == s[i]) {
      // If the current stone has the same color as the previous one,
      // it's a candidate for removal.
      count++;
    } else {
      // If the color is different, update the previous stone's color for the next iteration.
      a = s[i];
    }
  }
  // Print the total number of stones that need to be removed.
  cout << count;
  return 0;
}
