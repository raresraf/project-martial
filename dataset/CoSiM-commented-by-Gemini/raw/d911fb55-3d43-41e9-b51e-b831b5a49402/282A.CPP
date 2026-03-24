/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - "Bit++".
 * @details The program simulates a simple programming language with one variable, X,
 * initialized to 0. It executes a series of statements that either increment or
 * decrement X and outputs the final value of X.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. Read the number of statements, `n`.
 * 2. Initialize a `counter` variable to 0, representing X.
 * 3. Loop `n` times to process each statement.
 * 4. For each statement (read as a string), use a regular expression to check if
 *    it contains the substring "--".
 * 5. If the statement contains "--" (e.g., "--X" or "X--"), decrement the `counter`.
 * 6. Otherwise, the statement must contain "++" (e.g., "++X" or "X++"), so increment
 *    the `counter`.
 * 7. After all statements are processed, print the final value of the `counter`.
 *
 * @note The use of `std::regex` is an interesting but less performant choice for this
 * problem compared to simple character or substring searches.
 *
 * @complexity
 * Time Complexity: O(N * L), where N is the number of statements and L is the length
 * of each statement. The regex matching can take time proportional to the string length.
 * Since L is fixed at 3, the complexity is effectively O(N).
 * Space Complexity: O(L_max), for storing the command string.
 */
#include <iostream>
#include <regex>
#include <string>

int main() {
  int n;
  int counter = 0; // This variable represents 'X' in the problem.

  std::cin >> n; // Read the number of statements.

  /**
   * Block Logic: This loop processes each of the 'n' statements.
   */
  for (int i = 0; i < n; i++) {
    std::string command;
    std::cin >> command;

    /**
     * Block Logic: Determines whether to increment or decrement the counter.
     * It uses a regular expression to check if the command string contains "--".
     * If it does, it's a decrement operation. Otherwise, it's an increment.
     */
    if (std::regex_match(command, std::regex("(.*)(--)(.*)")))
      counter--;
    else
      counter++;
  }

  // Print the final value of the counter.
  std::cout << counter << std::endl;

  return 0;
}
