/**
 * @file 112A.CPP
 * @brief This program performs a case-insensitive lexicographical comparison of two input strings.
 * It converts both strings to lowercase and then compares them, outputting -1, 1, or 0 based on their order.
 * Algorithm: String normalization (to lowercase) followed by lexicographical comparison.
 * Time Complexity: O(L) where L is the length of the longer input string.
 * Space Complexity: O(L) for storing the input strings.
 */

#include <cstring> // Required for string manipulation, though not directly used in this implementation (string class handles length).
#include <iostream> // For input/output operations.
using namespace std;

/**
 * @brief Main function to compare two strings lexicographically in a case-insensitive manner.
 * Reads two strings, converts them to lowercase, and then prints -1, 1, or 0 based on their comparison.
 * @return 0 on successful execution.
 */
int main() {
  // @var a: First input string.
  // @var b: Second input string.
  string a, b;
  // Read two strings from standard input.
  cin >> a >> b;
  /**
   * @brief Normalize both strings to lowercase for case-insensitive comparison.
   * Invariant: After each iteration, characters up to index 'i' in both strings 'a' and 'b'
   * are converted to their lowercase equivalents if they were originally uppercase.
   */
  for (int i = 0; i < a.length(); i++) {
    // Check if the character in string 'a' is an uppercase letter (ASCII values 65-90).
    if (a[i] >= 65 && a[i] <= 90) {
      // Convert uppercase character to lowercase by adding 32 to its ASCII value.
      a[i] += 32;
    }
    // Check if the character in string 'b' is an uppercase letter.
    if (b[i] >= 65 && b[i] <= 90) {
      // Convert uppercase character to lowercase.
      b[i] += 32;
    }
  }
  // Perform lexicographical comparison on the normalized (lowercase) strings.
  // If 'a' is lexicographically smaller than 'b'.
  if (a < b)
    cout << "-1"; // Output -1.
  // If 'a' is lexicographically greater than 'b'.
  else if (a > b)
    cout << "1"; // Output 1.
  // If 'a' and 'b' are lexicographically equal.
  else if (a == b)
    cout << "0"; // Output 0.
  return 0;
}
