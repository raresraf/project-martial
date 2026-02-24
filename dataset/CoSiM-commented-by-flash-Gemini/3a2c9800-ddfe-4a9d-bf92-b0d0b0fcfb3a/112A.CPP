/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A - Petya and Strings.
 *
 * This program performs a case-insensitive lexicographical comparison of two input strings.
 * It determines which string comes first alphabetically or if they are identical,
 * outputting 1, -1, or 0 respectively (where 1 means the first string is greater,
 * -1 means it's smaller, and 0 means they are equal).
 *
 * Algorithm: Uppercase conversion and character-by-character comparison. Both strings
 * are first converted to uppercase. Then, they are compared character by character
 * from left to right. The comparison stops at the first differing character.
 * If all characters are the same and strings have equal length, they are considered equal.
 * Time Complexity: O(L), where L is the length of the strings, as each character
 *                  is processed twice (once for transform, once for comparison).
 * Space Complexity: O(L) for storing the input strings.
 */

#include <bits/stdc++.h> // Includes various standard libraries, including iostream, string, and algorithm.
using namespace std;     // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the Petya and Strings problem logic.
   *
   * Optimizes C++ standard stream I/O for faster execution.
   * Reads two strings from standard input. It then converts both strings
   * to uppercase to ensure case-insensitive comparison. Finally, it compares
   * the strings character by character and prints 1, -1, or 0 based on the
   * lexicographical comparison result.
   */
  // Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout.
  // This optimizes high-throughput I/O often required for competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Declare two string variables to hold the input lines.
  string a, b;
  // Read the two input strings.
  cin >> a >> b;
  
  // Convert string 'a' to uppercase.
  // std::transform applies a given operation to a range.
  // a.begin(), a.end() define the range. a.begin() is the output iterator.
  // ::toupper is the operation (converts char to uppercase).
  transform(a.begin(), a.end(), a.begin(), ::toupper);
  // Convert string 'b' to uppercase.
  transform(b.begin(), b.end(), b.begin(), ::toupper);

  // Declare 'x' to store the comparison result: 1 (a > b), -1 (a < b), 0 (a == b).
  int x = 0;
  // Block Logic: Character-by-character comparison of uppercase strings.
  // Invariant: 'i' iterates through the strings. The loop breaks upon the first difference.
  for (int i = 0; i < a.length(); i++) {
    // If character from string 'a' is greater than character from string 'b'.
    if ((int)a[i] > (int)b[i]) {
      x = 1;   // Set result to 1 (a is greater).
      break;   // Exit loop.
    }
    // If character from string 'a' is less than character from string 'b'.
    else if ((int)a[i] < (int)b[i]) {
      x = -1;  // Set result to -1 (a is smaller).
      break;   // Exit loop.
    }
    // If characters are equal, continue to the next iteration.
  }
  // Output the comparison result 'x', followed by a newline.
  cout << x << endl;
  return 0; // Indicate successful program execution.
}