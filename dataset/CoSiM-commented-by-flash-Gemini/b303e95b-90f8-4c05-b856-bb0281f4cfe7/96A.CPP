/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A: Football.
 * @details This program determines if a "dangerous" situation exists in a football game,
 * characterized by a sequence of 7 or more consecutive players of the same team.
 * The input is a string of '0's and '1's, representing players of two teams.
 * The program iterates through the string to find any substring of 7 or more identical characters.
 * @algorithm String traversal and pattern matching.
 * Time Complexity: $O(L)$ where L is the length of the input string, as it iterates through the string once.
 * Space Complexity: $O(L)$ to store the input string.
 */
#include <iostream>
#include <string>
using namespace std;
/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A: Football.
 * @details This program determines if a "dangerous" situation exists in a football game,
 * characterized by a sequence of 7 or more consecutive players of the same team.
 * The input is a string of '0's and '1's, representing players of two teams.
 * The program iterates through the string to find any substring of 7 or more identical characters.
 * @algorithm String traversal and pattern matching.
 * Time Complexity: $O(L)$ where L is the length of the input string, as it iterates through the string once.
 * Space Complexity: $O(L)$ to store the input string.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <string>   // Required for string manipulation (std::string, size, character access).
using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the "dangerous situation" detection logic.
   */
  string s; // @var s: The input string representing the sequence of players.
  int i,    // @var i: Outer loop counter, iterating through possible start positions of a dangerous sequence.
      j;    // @var j: Inner loop counter, checking for consecutive identical characters.

  cin >> s; // Reads the input string from standard input.

  /**
   * @brief Iterates through the string to find a sequence of 7 or more identical characters.
   * @block_logic Outer loop attempts to find a potential start of a dangerous sequence.
   * @invariant `i` represents the starting index of a potential sequence.
   */
  for (i = 0; i < s.size(); i++) {
    /**
     * @brief Checks if the character at `s[i]` is the same as `s[i+6]`.
     * @details This initial check efficiently prunes non-dangerous sequences. If `s[i]` is not
     *          equal to `s[i+6]`, then a sequence of 7 identical characters starting at `s[i]`
     *          is impossible, and the inner loop is skipped.
     */
    if (s[i] == s[i + 6]) { // Checks if `s[i]` and `s[i+6]` are the same (implying 7 elements if intermediate are same).
      /**
       * @brief Inner loop to verify 7 consecutive identical characters.
       * @block_logic Iterates from `s[i+1]` to `s[i+5]` to confirm all are identical to `s[i]`.
       * @invariant If `s[i+j]` is different from `s[i]`, the sequence is broken, and the inner loop breaks.
       */
      for (j = 1; j < 6; j++) { // Checks positions `i+1` through `i+5`.
        if (j == 5 && s[i + j] == s[i]) { // If `j` reaches 5 and `s[i+5]` is also `s[i]`, then all 7 are same.
          cout << "YES"; // Output "YES" (dangerous situation detected).
          return 0;      // Terminate the program.
        } else {
          if (s[i + j] == s[i]) // If current character matches `s[i]`, continue checking.
            continue;
          else
            break; // If a mismatch, break the inner loop.
        }
      }
    }
  }
  // Functional Utility: If no dangerous situation is found after checking all possible start positions.
  cout << "NO"; // Output "NO".
  return 0;     // Indicates successful execution.
}
