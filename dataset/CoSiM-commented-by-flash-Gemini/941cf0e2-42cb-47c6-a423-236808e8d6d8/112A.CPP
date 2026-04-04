/**
 * @file 112A.CPP
 * @brief Lexicographical string comparison, ignoring case, with manual lowercase conversion.
 *
 * This program reads two strings (potentially containing spaces) from standard input,
 * manually converts all their characters to lowercase based on ASCII values, and then
 * compares them lexicographically. It outputs 1 if the first string is larger,
 * -1 if it's smaller, and 0 if they are equal.
 *
 * Algorithm: Manual character-by-character case conversion followed by standard string comparison.
 * Time Complexity: O(L) where L is the length of the longer string, as both manual conversion
 *                  and comparison operations are linear.
 * Space Complexity: O(1) beyond input string storage.
 */
#include <bits/stdc++.h> // Includes most standard library headers, including <string> and <iostream>.

using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: Reads two strings, converts them to lowercase manually,
   *        compares them lexicographically, and prints the result.
   * @return 0 for successful execution.
   */
  string s1, s2; // Declare two strings for input.
  getline(cin, s1); // Read the first entire line (string) from standard input.
  getline(cin, s2); // Read the second entire line (string) from standard input.

  /**
   * @brief Loop to manually convert all uppercase characters in both strings to lowercase.
   * @pre 's1' and 's2' contain the input strings.
   * @post 's1' and 's2' contain their lowercase equivalents.
   * @invariant 'i' iterates through the length of 's1'. Assumes 's1' and 's2' have equal length or 's1' is shorter than 's2'.
   */
  for (int i = 0; i < s1.size(); i++) {
    // Block Logic: Converts uppercase ASCII characters (65-91) to lowercase by adding 32.
    if (s1[i] >= 65 && s1[i] <= 91) {
      s1[i] = s1[i] + 32;
    } else {
      s1[i] = s1[i]; // Character is already lowercase or not an alphabet, keep as is.
    }

    // Block Logic: Converts uppercase ASCII characters in s2 to lowercase.
    if (s2[i] >= 65 && s2[i] <= 91) {
      s2[i] = s2[i] + 32;
    } else {
      s2[i] = s2[i]; // Character is already lowercase or not an alphabet, keep as is.
    }
  }

  /**
   * @brief Conditional block to print the lexicographical comparison result.
   * @pre 's1' and 's2' are converted to lowercase.
   * @post Prints 1, -1, or 0 and terminates the program.
   */
  if (s1.compare(s2) > 0) { // If s1 is lexicographically greater than s2.
    printf("1\n"); // Output 1.
  } else if (s1.compare(s2) < 0) { // If s1 is lexicographically smaller than s2.
    printf("-1\n"); // Output -1.
  } else { // If s1 and s2 are lexicographically equal.
    printf("0\n"); // Output 0.
  }

  return 0; // Return 0 for successful execution.
}
