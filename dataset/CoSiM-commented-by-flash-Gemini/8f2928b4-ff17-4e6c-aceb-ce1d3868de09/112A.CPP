/**
 * @file 112A.CPP
 * @brief Lexicographical string comparison, ignoring case.
 *
 * This program reads two strings, converts them to lowercase, and then
 * compares them lexicographically. It outputs -1 if the first string is
 * lexicographically smaller, 1 if it's larger, and 0 if they are equal.
 *
 * Algorithm: String iteration with character-by-character comparison after
 *            converting characters to lowercase.
 * Time Complexity: O(min(L1, L2)), where L1 and L2 are the lengths of the two strings.
 *                  The `transform` operations each take O(L) time.
 * Space Complexity: O(1) beyond input storage, as `transform` operates in-place.
 */
#include <algorithm>
#include <iostream>

using namespace std;

// Functional Utility: Defines 'll' as a shorthand for 'long long' for convenience.
#define long long ll;

// Global variables to store the two input strings for comparison.
string s1, s2;

void solve() {
  /**
   * @brief Compares two global strings (s1 and s2) lexicographically, ignoring case.
   *
   * Iterates through the strings character by character, converting them to
   * their integer ASCII values for comparison. Sets a global 'j' to 1 if s1 > s2,
   * -1 if s1 < s2, and leaves it as 0 if s1 == s2. The first differing character
   * determines the result.
   *
   * @pre s1 and s2 contain the input strings, already converted to lowercase.
   * @post Global variable 'j' is set to 1, -1, or 0 based on comparison.
   */
  int i = 0, j = 0; // 'i': current character index, 'j': comparison result (1, -1, or 0).
  bool found = false; // Flag to indicate if a difference has been found.
  /**
   * @brief Loop to compare characters of s1 and s2 until a difference is found or end of strings.
   * @invariant 'i' increments through the strings; 'j' remains 0 until a difference is found.
   */
  while (i < s1.length() && i < s2.length() && !found) {
    int c1 = int(s1[i]); // Get ASCII value of character from s1.
    int c2 = int(s2[i]); // Get ASCII value of character from s2.
    /**
     * @brief Checks if current characters differ.
     */
    if (c1 != c2) {
      // Calculate the comparison result: 1 if s1[i] > s2[i], -1 if s1[i] < s2[i].
      // Inline: abs(c1 - c2) prevents division by zero if c1 == c2 (which is already handled by if condition).
      j = (c1 - c2) / abs(c1 - c2); 
      found = true; // Mark that a difference has been found.
    }
    i++; // Move to the next character.
  }
  cout << j; // Output the comparison result.
}

int main() {
  /**
   * @brief Main function: Reads two strings, converts them to lowercase, and compares them.
   * @return 0 if the program executes successfully.
   */
  cin >> s1 >> s2; // Read the two input strings.
  // Functional Utility: Converts all characters in s1 to lowercase in-place.
  transform(s1.begin(), s1.end(), s1.begin(),
            [](unsigned char c) -> unsigned char { return tolower(c); });
  // Functional Utility: Converts all characters in s2 to lowercase in-place.
  transform(s2.begin(), s2.end(), s2.begin(),
            [](unsigned char c) -> unsigned char { return tolower(c); });
  solve(); // Call the solve function to perform the comparison and print the result.
  return 0;
}