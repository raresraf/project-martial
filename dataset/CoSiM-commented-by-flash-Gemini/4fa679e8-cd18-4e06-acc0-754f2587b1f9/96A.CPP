/**
 * @file 96A.CPP
 * @brief Checks for "dangerous situations" in a sequence of players, defined as seven consecutive identical players.
 *
 * Problem Name: Football (Codeforces 96A)
 * Algorithm: Iterative scanning. It maintains a count of consecutive identical characters and stops
 *            as soon as a sequence of seven is found.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for the input string, and O(1) for auxiliary variables.
 */
#include <iostream> // Required for std::cin, std::cout.
#include <string>   // Required for std::string.

/**
 * @brief Main function to detect dangerous situations in a player sequence string.
 * @pre Input consists of a single string of '0's and '1's.
 * @post Prints "YES" if a dangerous situation is found (seven consecutive identical characters), otherwise "NO".
 */
int main() {
  std::string s; // Variable to store the input string representing player sequence.
  // Functional Utility: Reads the input string from standard input.
  std::cin >> s;

  auto count = 1;      // Counter for consecutive identical characters.
  auto temp = ' ';     // Stores the previous character for comparison. Initialized to a non-matching character.

  /**
   * Block Logic: Iterates through each character of the string to find consecutive identical characters.
   * Invariant: `count` tracks the length of the current sequence of identical characters.
   *            If `count` reaches 7, the loop terminates early.
   */
  for (auto letter : s) { // Range-based for loop for iterating over characters in string 's'.
    /**
     * Block Logic: Updates the consecutive count based on the current and previous character.
     * Pre-condition: `letter` is the current character, `temp` is the previous character.
     * Invariant: If `letter` matches `temp`, `count` increases. Otherwise, `count` resets to 1.
     */
    if (temp == letter) {
      count++; // Increments count of consecutive identical characters.
    } else {
      count = 1; // Resets count if character changes.
    }
    temp = letter; // Updates `temp` to the current character for the next iteration.

    /**
     * Block Logic: Early exit condition if a dangerous situation is detected.
     * Pre-condition: `count` reflects the current number of consecutive identical characters.
     * Invariant: If `count` is 7, the dangerous situation is confirmed, and further scanning is unnecessary.
     */
    if (count == 7) {
      break; // Exits the loop early.
    }
  }

  /**
   * Block Logic: Prints the final result based on whether a dangerous situation was found.
   * Pre-condition: `count` holds the highest consecutive count (or 7 if found).
   * Invariant: If `count` is 7, it means a sequence of seven identical characters was found.
   */
  if (count == 7) {
    std::cout << "YES" << std::endl; // Prints "YES" if dangerous.
  } else {
    std::cout << "NO" << std::endl;  // Prints "NO" otherwise.
  }
  // Indicates successful program execution.
  return 0;
}
