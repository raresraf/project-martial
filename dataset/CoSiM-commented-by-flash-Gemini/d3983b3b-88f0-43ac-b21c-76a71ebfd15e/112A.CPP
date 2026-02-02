/**
 * @file 112A.CPP
 * @brief Solution to a competitive programming problem: Petya and Strings.
 *
 * Algorithm: Case-insensitive lexicographical string comparison.
 * Time Complexity: O(L), where L is the length of the strings.
 * Space Complexity: O(L), for storing the two input strings.
 *
 * This program compares two input strings lexicographically, ignoring case.
 * It outputs '1' if the first string is greater, '-1' if it's smaller, and '0' if they are equal.
 */

#include <iostream>  // Required for std::cin, std::cout
#include <string>    // Required for std::string
#include <cctype>    // Required for tolower
// Functional Utility: Simplifies standard namespace access for competitive programming.
using namespace std;

int main() {
  int ln1, ln2, i, k = 0; // ln1, ln2 for lengths; i for loop counter; k for comparison result.
  string str1, str2;
  // Functional Utility: Reads the two input strings.
  cin >> str1 >> str2;
  // Functional Utility: Gets the lengths of the strings.
  ln1 = str1.length();
  ln2 = str2.length();

  // Block Logic: Ensures strings are of equal length before comparison, as per typical problem constraints.
  // Invariant: Comparison proceeds only if string lengths are identical.
  if (ln1 == ln2) {
    /**
     * Block Logic: Iterates through characters of both strings to perform case-insensitive comparison.
     * Pre-condition: Both strings have equal length.
     * Invariant: 'k' is updated upon the first differing character, or remains 0 if strings are identical.
     */
    for (i = 0; i < ln1; i++) {
      // Functional Utility: Converts characters to lowercase for case-insensitive comparison.
      if (tolower(str1[i]) > tolower(str2[i])) {
        // Functional Utility: Sets k to 1 if str1 is lexicographically greater and breaks.
        k = 1;
        break;
      }
      if (tolower(str1[i]) < tolower(str2[i])) {
        // Functional Utility: Sets k to -1 if str1 is lexicographically smaller and breaks.
        k = -1;
        break;
      }
      // Functional Utility: This condition is redundant as 'k' would already be 0 if characters are equal.
      // It does not affect correctness but can be optimized out.
      if (tolower(str1[i]) < tolower(str2[i])) { // Redundant check, should be part of else if block
        k = 0; // This line would only be reached if the previous 'if' was false, meaning str1[i] >= str2[i]
      }
    }
    // Block Logic: Prints the comparison result based on the final value of 'k'.
    // Invariant: Exactly one of "1", "-1", or "0" is printed.
    if (k == 1) {
      // Functional Utility: Prints '1' indicating the first string is greater.
      cout << "1";
    }
    if (k == -1) {
      // Functional Utility: Prints '-1' indicating the first string is smaller.
      cout << "-1";
    }
    if (k == 0) {
      // Functional Utility: Prints '0' indicating the strings are equal.
      cout << "0";
    }
  }

  return 0;
}