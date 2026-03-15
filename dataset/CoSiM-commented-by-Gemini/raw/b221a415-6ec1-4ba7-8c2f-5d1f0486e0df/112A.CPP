/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem (case-insensitive string comparison).
 * @details This program reads two strings and compares them lexicographically, ignoring case.
 * It prints -1, 1, or 0 depending on whether the first string is less than,
 * greater than, or equal to the second.
 *
 * @algorithm
 * 1. Read two strings from standard input.
 * 2. Use the `std::transform` algorithm to apply the `::tolower` function to every
 *    character in both strings, making the comparison case-insensitive.
 * 3. Manually iterate through the strings character by character to perform the
 *    lexicographical comparison.
 * 4. At the first differing character, print -1 or 1 and break the loop.
 * 5. If the loop finishes without finding any differences, a flag is checked,
 *    and 0 is printed to indicate the strings are equal.
 *
 * @complexity
 * - Time: O(L), where L is the length of the strings. The `transform` operations and the
 *   comparison loop each take linear time.
 * - Space: O(L) to store the two input strings.
 */
#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s1, s2;
  cin >> s1;
  cin >> s2;

  // Use std::transform to convert both strings to lowercase in-place.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  // A flag to track if a lexicographical difference has been found.
  bool check = false;

  /**
   * @brief This loop performs a manual lexicographical comparison.
   * It iterates through the strings and compares them character by character.
   */
  for (int i = 0; i < s1.length(); i++) {
    if (s1[i] < s2[i]) {
      cout << "-1";
      check = true; // Mark that a difference was found.
      break;
    } else if (s1[i] == s2[i]) {
      // If characters are the same, continue to the next pair.
      continue;
    } else {
      cout << "1";
      check = true; // Mark that a difference was found.
      break;
    }
  }

  // Pre-condition: If `check` is false, the loop completed without finding any differences.
  // Post-condition: The strings must be equal, so print 0.
  if (!check) {
    cout << "0";
  }
  return 0;
}