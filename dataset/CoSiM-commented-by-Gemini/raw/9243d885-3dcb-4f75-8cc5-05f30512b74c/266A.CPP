/**
 * @file 266A.CPP
 * @brief Counts the number of adjacent identical characters in a string.
 *
 * This program solves a classic introductory problem often framed as "stones on a table".
 * It calculates the minimum number of stones to remove from a row such that no two
 * adjacent stones have the same color. This is equivalent to counting the number of
 * times a character is the same as the one immediately preceding it.
 *
 * @algorithm Single-pass linear scan.
 * @complexity
 * - Time: O(N), where N is the length of the string, as it iterates through the string once.
 * - Space: O(N) to store the input string.
 */
#include <iostream>

using namespace std;

int main() {
  string stone;
  // `count` will store the number of stones to remove.
  int count = 0;
  // `n` is the total number of stones.
  int n;
  cin >> n >> stone;

  /**
   * @brief Block Logic: Iterates through the string to find adjacent identical characters.
   * Invariant: At each step `i`, `count` holds the number of adjacent identical
   *            characters found in the prefix of the string of length `i`.
   */
  for (int i = 1; i < n; i++) {
    // Pre-condition: Check if the current character is identical to the previous one.
    if (stone[i] == stone[i - 1])
      count++;
  }

  // Output the total number of characters that need to be "removed".
  cout << count << endl;
  return 0;
}
