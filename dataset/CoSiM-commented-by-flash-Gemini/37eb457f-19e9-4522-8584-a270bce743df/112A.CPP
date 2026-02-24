/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A - Petya and Strings.
 *
 * This program performs a case-insensitive lexicographical comparison of two input strings.
 * It determines which string comes first alphabetically or if they are identical,
 * outputting -1, 1, or 0 respectively.
 *
 * Algorithm: Character-by-character comparison. The strings are compared character by character
 * from left to right. Before comparison, each character is converted to lowercase to ensure
 * case-insensitivity. The comparison stops at the first differing character or if both strings
 * are fully traversed and found to be equal.
 * Time Complexity: O(L), where L is the length of the shorter string (or either string if lengths are equal).
 * Space Complexity: O(1), excluding input storage.
 */

#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For string manipulation.
#include <cctype>   // For tolower() function.

// Using declarations for commonly used standard library components to avoid std:: prefix.
using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  /**
   * @brief Main function to execute the Petya and Strings problem logic.
   *
   * Reads two strings from standard input. It then compares them lexicographically
   * in a case-insensitive manner. Based on the comparison, it prints -1, 1, or 0
   * to standard output and terminates.
   */
  // Declare two string variables to hold the input lines.
  string line1, line2;
  // Read the two input strings.
  cin >> line1 >> line2;

  // Block Logic: Basic check for string length equality.
  // Although not strictly necessary for lexicographical comparison if one string
  // is a prefix of another, for this problem's typical constraints, unequal lengths
  // might indicate an invalid input or a problem variant.
  // (Note: In standard lexicographical comparison, if one string is a prefix of another,
  // the shorter one is considered smaller. This specific implementation returns -1 for unequal lengths.)
  if (line1.size() != line2.size()) {
    // If lengths are not equal, this implementation terminates with an error code.
    // A standard competitive programming problem would likely handle this to mean
    // the shorter string is lexicographically smaller if it's a prefix of the longer one.
    return -1; // Indicates an issue or specific problem constraint.
  }

  // Block Logic: Character-by-character case-insensitive comparison.
  // Invariant: 'i' iterates from the beginning of the strings.
  // Pre-condition: Strings are assumed to have equal length at this point.
  for (decltype(line1.size()) i = 0; i != line1.size(); ++i) {
    // Compare characters after converting both to lowercase.
    // If character from line1 is lexicographically smaller than character from line2.
    if (tolower(line1[i]) < tolower(line2[i])) {
      cout << "-1"; // Output -1, indicating line1 is smaller.
      return 0;     // Terminate program successfully.
    }
    // If character from line1 is lexicographically greater than character from line2.
    else if (tolower(line1[i]) > tolower(line2[i])) {
      cout << "1";  // Output 1, indicating line1 is larger.
      return 0;     // Terminate program successfully.
    }
    // If characters are equal, continue to the next iteration (next pair of characters).
  }

  // If the loop completes, it means all characters were identical.
  cout << "0";    // Output 0, indicating strings are equal.
  return 0;       // Terminate program successfully.
}
