// 266A
/**
 * @file 266A.CPP
 * @brief This program calculates the number of adjacent identical characters in a given string.
 *
 * The program reads an integer N, representing the length of a string, and then reads the string itself.
 * It iterates through the string, comparing each character with the previous one. If a character is identical
 * to its preceding character, a counter is incremented. Finally, it prints the total count of such occurrences.
 *
 * Algorithm: Linear scan with state-tracking.
 * Time Complexity: O(L) where L is the length of the input string, as it performs a single pass.
 * Space Complexity: O(1) for character storage and counter.
 */
#include <cstring> // Functional Utility: Provides string manipulation functions, although not explicitly used for complex operations here.
#include <iostream> // Functional Utility: Manages input and output streams.
using namespace std;

int main() {
  int n; // Variable: Stores the length of the input string.
  string s; // Variable: Stores the input string.

  /**
   * Block Logic: Reads the length of the string and the string itself from standard input.
   * Pre-condition: Standard input must contain an integer followed by a string.
   */
  cin >> n;
  cin >> s;

  char a; // Variable: Stores the previously encountered character for comparison.
  int count = 0; // Variable: Accumulates the count of adjacent identical characters.
  a = s[0]; // Initialization: Sets the first character of the string as the initial reference for comparison.

  /**
   * Block Logic: Iterates through the string from the second character to detect adjacent duplicates.
   * Invariant: At the start of each iteration, 'a' holds the character from the previous position (i-1),
   * and 'count' reflects the number of adjacent duplicates found up to position (i-1).
   */
  for (int i = 1; i < s.length(); i++) {
    /**
     * Block Logic: Checks if the current character is identical to the previous one.
     * Pre-condition: 'a' holds the character at index (i-1), 's[i]' is the current character.
     */
    if (a == s[i]) {
      count++; // Action: Increments the counter if a duplicate is found.
    } else {
      a = s[i]; // State Update: Updates the reference character if the current character is different.
    }
  }

  // Output: Prints the total count of adjacent identical characters.
  cout << count;
  return 0;
}