/**
 * @file 58A.CPP
 * @brief Checks if the word "hello" can be formed as a subsequence from an input string.
 *
 * This program reads an input string and determines if the characters 'h', 'e', 'l', 'l', 'o'
 * appear in order within the string, allowing for zero or more characters to be skipped
 * between them. This is a classic subsequence matching problem, often found in
 * competitive programming challenges.
 *
 * Algorithm: Two-pointer approach. One pointer (`i`) iterates through the input string,
 *            and another pointer (`j`) tracks the progress of matching the "hello" sequence.
 * Time Complexity: O(N), where N is the length of the input string, as it involves a single pass.
 * Space Complexity: O(1) for storing constant-size strings and counters.
 */
// Codeforces
// 58A - Chat room

#include <iostream> // Required for standard input/output operations.
using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: Reads an input string and checks for "hello" as a subsequence.
   * @return 0 if the program executes successfully.
   */
  string s; // Input string.
  cin >> s; // Read the input string.

  unsigned int i = 0, j = 0; // 'i': pointer for input string 's', 'j': pointer for target string 'g'.

  string g = "hello"; // Target subsequence.
  string output = "NO"; // Default output.

  /**
   * @brief Loop to iterate through the input string 's' and find the "hello" subsequence.
   * @pre 's' contains the input string; 'g' is the target "hello"; 'i' and 'j' are initialized to 0.
   * @post If "hello" is found, 'output' will be "YES" and the loop will break.
   * @invariant 'i' advances through 's'; 'j' advances only when a character from 'g' is matched.
   */
  while (i < s.size()) {
    /**
     * @brief Checks if the current character in 's' matches the next expected character in 'g'.
     * @post If a match, 'j' is incremented. 'i' is always incremented.
     */
    if (s[i++] == g[j]) { // Inline: 'i' is incremented after comparison to move to the next character in 's'.
      j++; // Advance pointer for 'g' if character matches.
    }

    /**
     * @brief Checks if all characters of "hello" have been found.
     * @pre 'j' tracks the number of matched characters. 'g.size()' is 5 for "hello".
     */
    if (j >= g.size()) { // If all characters of "hello" are matched.
      output = "YES";   // Set output to "YES".
      break;            // Exit the loop as subsequence is found.
    }
  }

  cout << output << endl; // Output the result ("YES" or "NO").

  return 0; // Return 0 for successful execution.
}
