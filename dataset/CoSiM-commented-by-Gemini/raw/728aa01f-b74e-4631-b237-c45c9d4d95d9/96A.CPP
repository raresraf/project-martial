/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem detecting "dangerous" patterns in a string.
 * This program reads a string representing a sequence of plays and determines if there are
 * 7 or more consecutive identical characters, which is defined as a "dangerous" situation.
 *
 * Algorithm: String iteration with consecutive character counting.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <iostream>
#include <string>

/**
 * @brief Main function of the program.
 * Reads an input string `str` and checks for the presence of 7 or more
 * consecutive identical characters. It then prints "YES" if such a pattern
 * is found, and "NO" otherwise.
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return `0` upon successful execution.
 */
int main(int argc, char *argv[]) {
  std::string str; // Declares a string to store the input sequence.
  std::cin >> str; // Reads the input string from standard input.

  bool dangerous = false; // Flag to indicate if a dangerous pattern has been found.
  int conseq = 1;         // Counter for consecutive identical characters.
  /**
   * @brief Loop to iterate through the string and count consecutive identical characters.
   * Invariant: `i` iterates from the second character to the end of the string.
   * `conseq` tracks the current count of consecutive identical characters.
   */
  for (size_t i = 1; i < str.size(); ++i) {
    /**
     * @brief Conditional logic for resetting or incrementing the consecutive counter.
     * If the current character is different from the previous one, `conseq` is reset to 1.
     * Otherwise, `conseq` is incremented.
     */
    if (str[i - 1] != str[i]) {
      conseq = 1;
      continue;
    }

    conseq++; // Increment consecutive count if characters are identical.
    /**
     * @brief Checks if a "dangerous" pattern (7 consecutive identical characters) is found.
     * If `conseq` reaches 7, the `dangerous` flag is set to true, and the loop terminates early.
     */
    if (conseq == 7) {
      dangerous = true;
      break;
    }
  }

  // Prints "YES" if a dangerous pattern was found, otherwise "NO".
  std::cout << (dangerous ? "YES" : "NO") << std::endl;
  return 0;
}
