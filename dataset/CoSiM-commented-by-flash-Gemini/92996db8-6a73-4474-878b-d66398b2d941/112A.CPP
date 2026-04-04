/**
 * @file 112A.CPP
 * @brief Lexicographical string comparison, ignoring case, using character-by-character comparison.
 *
 * This program reads two strings and compares them lexicographically, ignoring case differences.
 * It outputs 1 if the first string is lexicographically larger, -1 if it's smaller,
 * and 0 if they are equal. The comparison proceeds character by character until a difference
 * is found or the end of the shorter string is reached.
 *
 * Algorithm: Character-by-character comparison after converting to lowercase.
 * Time Complexity: O(L) where L is the length of the shorter string, as the loop terminates
 *                  on the first differing character or the end of either string.
 * Space Complexity: O(1) beyond input storage.
 */
#include <bits/stdc++.h> // Includes most standard library headers, including <string> and <iostream>.
using namespace std;     // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: Reads two strings, compares them lexicographically ignoring case, and prints the result.
   * @return 0 for successful execution (program terminates immediately after printing).
   */
  string a, b; // Declare two strings for input.
  cin >> a >> b; // Read the two input strings.
  int len = a.length(); // Get the length of the first string (assuming both strings have similar lengths for primary comparison).
  /**
   * @brief Loop to compare characters of strings 'a' and 'b' one by one, ignoring case.
   * @pre 'a' and 'b' contain the input strings.
   * @post Prints 1, -1, or 0 and terminates the program upon finding a difference or reaching the end.
   * @invariant 'i' iterates through the strings; comparison occurs character by character.
   */
  for (int i = 0; i < len; i++) {
    /**
     * @brief Checks if the character in string 'a' is lexicographically greater than in 'b', ignoring case.
     */
    if ((int)tolower(a[i]) > (int)tolower(b[i])) { // Convert to lowercase and compare ASCII values.
      cout << 1;     // Output 1 if 'a' is greater.
      return 0;      // Terminate program.
    }
    /**
     * @brief Checks if the character in string 'a' is lexicographically less than in 'b', ignoring case.
     */
    if ((int)tolower(a[i]) < (int)tolower(b[i])) { // Convert to lowercase and compare ASCII values.
      cout << -1;    // Output -1 if 'a' is smaller.
      return 0;      // Terminate program.
    }
  }

  cout << 0; // If the loop completes, strings are equal (or one is a prefix of another of equal length).
             // Since the loop condition uses a.length(), this implies they are equal if no difference is found.
  // No explicit return 0 here, but main function implicitly returns 0 on reaching end.
}
