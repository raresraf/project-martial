/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem using manual ASCII conversion for case-insensitivity.
 * @details This program compares two strings lexicographically. It normalizes both strings
 * to lowercase using direct ASCII value manipulation before performing the final comparison.
 *
 * @algorithm The algorithm first converts both input strings to lowercase in-place. It iterates
 * through the strings, character by character, and checks if a character's ASCII value falls
 * within the range for uppercase English letters (65-90). If it does, 32 is added to the value
 * to convert it to its lowercase counterpart. After this normalization pass, the standard
 * `std::string` comparison operators are used to determine the result (0, -1, or 1).
 *
 * @time_complexity O(L), where L is the length of the strings, due to the linear scan for
 * normalization and the subsequent linear-time comparison.
 * @space_complexity O(L), required to store the two input strings.
 */
#include <iostream>

using namespace std;

int main() {
  string a;
  string b;
  cin >> a;
  cin >> b;

  /**
   * Block Logic: Normalize both strings to lowercase in a single pass using ASCII arithmetic.
   */
  for (int i = 0; i < a.length(); i++) {
    // Inline: Check if the character is within the ASCII range for uppercase 'A' through 'Z' (65-90).
    if (a[i] > 64 && a[i] < 91) {
      // Convert to lowercase by adding 32 (the ASCII difference between 'a' and 'A').
      a[i] += 32;
    }

    if (b[i] > 64 && b[i] < 91) {
      b[i] += 32;
    }
  }

  /**
   * Block Logic: Perform lexicographical comparison on the now-lowercase strings.
   */
  if (a < b) {
    cout << -1 << "\n";
  } else if (a > b) {
    cout << 1 << "\n";
  } else if (a == b) {
    cout << 0 << "\n";
  }
  return 0;
}