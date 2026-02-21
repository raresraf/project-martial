/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings from standard input and compares them
 * lexicographically, ignoring case. It prints -1 if the first string is
 * lexicographically smaller, 1 if it is larger, and 0 if they are equal.
 *
 * Algorithm:
 * 1. Read two strings, `line1` and `line2`.
 * 2. Iterate through both strings from the beginning to the end.
 * 3. In each iteration, convert the characters at the current position to lowercase.
 * 4. Compare the lowercase characters.
 *    - If `line1[i]` is less than `line2[i]`, the first string is smaller. Print -1 and terminate.
 *    - If `line1[i]` is greater than `line2[i]`, the first string is larger. Print 1 and terminate.
 * 5. If the loop finishes without any character difference, the strings are equal. Print 0.
 *
 * Time Complexity: O(N), where N is the length of the strings, as it involves a single pass.
 * Space Complexity: O(N) to store the two input strings.
 */
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string line1, line2;
  // Read the two strings to be compared.
  cin >> line1 >> line2;

  // This check is redundant for this specific problem's constraints but is good practice.
  if (line1.size() != line2.size()) {
    return -1;
  }

  /**
   * @brief Perform a character-by-character comparison in a case-insensitive manner.
   * The loop iterates through the strings and compares the lowercase versions of each character.
   */
  for (decltype(line1.size()) i = 0; i != line1.size(); ++i) {
    // Pre-condition: Characters at index `i` in both strings are being compared.
    if (tolower(line1[i]) < tolower(line2[i])) {
      cout << "-1";
      return 0;
    } else if (tolower(line1[i]) > tolower(line2[i])) {
      cout << "1";
      return 0;
    }
    // Invariant: The substrings from index 0 to `i` are lexicographically equal (case-insensitively).
  }

  // If the loop completes, the strings are equal.
  cout << "0";
  return 0;
}
