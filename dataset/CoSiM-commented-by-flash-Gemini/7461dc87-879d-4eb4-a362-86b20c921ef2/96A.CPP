/**
 * @file 96A.CPP
 * @brief Solution to Codeforces Problem 96A: "Football".
 *
 * Determines if a "dangerous situation" occurs in a football game, defined as
 * having 7 or more consecutive players of the same team (represented by '0' or '1')
 * in a sequence.
 *
 * Algorithm: Iterate through the player sequence, maintaining counters for consecutive
 * '0's and '1's. If either counter reaches 7, a dangerous situation is found.
 *
 * Time Complexity: O(L), where L is the length of the input string, as it involves
 *                  a single pass through the string.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream> // Required for `std::cin`, `std::cout`.
#include <string>   // Required for `std::string`.
using namespace std;

/**
 * @brief Main function to solve the "Football" problem.
 * Reads a string representing a sequence of players and determines if it contains
 * a dangerous situation (7 or more consecutive identical players).
 */
int main() {
  int nb0 = 0; // Counter for consecutive '0's.
  int nb1 = 0; // Counter for consecutive '1's.
  
  std::string s; // Input string representing the sequence of players.
  std::cin >> s; // Reads the input string.

  /**
   * Block Logic: Iterates through the string `s` to detect sequences of identical characters.
   * Pre-condition: `s` contains the player sequence. `nb0` and `nb1` are 0.
   * Invariant: `nb0` and `nb1` accurately track the current streak of '0's and '1's.
   *
   * Note on loop condition: `i <= s.size()` means `s[s.size()]` will be accessed.
   * For `std::string`, `s.size()` is one past the last valid index. Accessing `s[s.size()]`
   * results in undefined behavior. However, this pattern is sometimes used in competitive
   * programming, implicitly relying on `s[s.size()]` to be a null terminator in some contexts,
   * or to trigger the logic one last time for the final streak. This solution relies on
   * `s[s.size()]` not being '0' or '1' to reset the counters, correctly processing the last streak.
   */
  for (int i = 0; i <= s.size(); i++) {
    /**
     * Block Logic: If the current character is '1', increment `nb1` and reset `nb0`.
     */
    if (s[i] == '1') {
      nb1++;   // Increment count of consecutive '1's.
      nb0 = 0; // Reset count of consecutive '0's.
    }
    /**
     * Block Logic: If the current character is '0', increment `nb0` and reset `nb1`.
     */
    if (s[i] == '0') {
      nb1 = 0;   // Reset count of consecutive '1's.
      nb0++;   // Increment count of consecutive '0's.
    }
    /**
     * Block Logic: Check if a dangerous situation (7 or more consecutive identical players) has occurred.
     * If either `nb0` or `nb1` reaches 7 or more, print "YES" and terminate.
     */
    if (nb0 >= 7 || nb1 >= 7) {
      std::cout << "YES"; // Dangerous situation found.
      return 0;           // Terminate the program early.
    }
  }
  // If the loop completes without finding a dangerous situation, print "NO".
  std::cout << "NO";
  return 0; // Indicates successful execution.
}
