/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A: Team (Alternative Implementation).
 * @details This program processes 'n' problem statements, each with three opinions (represented by integers x, y, z, typically 0 or 1).
 * It determines how many problems will be implemented. A problem is implemented if at least two people agree
 * that it should be implemented, which is robustly checked by summing the opinions and verifying if the sum is 2 or more.
 * This is an alternative, more direct approach compared to using bitwise operations for boolean checks.
 * @algorithm Counting based on summation and thresholding.
 * Time Complexity: $O(N)$ where N is the number of problem statements.
 * Space Complexity: $O(1)$.
 */
#include <iostream>

using namespace std;

/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A: Team (Alternative Implementation).
 * @details This program processes 'n' problem statements, each with three opinions (represented by integers x, y, z, typically 0 or 1).
 * It determines how many problems will be implemented. A problem is implemented if at least two people agree
 * that it should be implemented, which is robustly checked by summing the opinions and verifying if the sum is 2 or more.
 * This is an alternative, more direct approach compared to using bitwise operations for boolean checks.
 * @algorithm Counting based on summation and thresholding.
 * Time Complexity: $O(N)$ where N is the number of problem statements.
 * Space Complexity: $O(1)$.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).

using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the problem counting logic.
   */
  int test, // @var test: The total number of problem statements to process.
      count = 0; // @var count: Counter for the number of problems that will be implemented.

  std::cin >> test; // Reads the total number of problem statements.

  /**
   * @brief Processes each of the 'test' problem statements.
   * @block_logic Iterates `test` times, processing one problem statement in each iteration.
   * @invariant `test` decreases by 1 in each iteration until it becomes 0.
   */
  while (test--) {
    int x, y, z; // @var x, y, z: Integers representing the opinions of three individuals for a problem (typically 0 or 1).
    std::cin >> x >> y >> z; // Reads the three opinions for the current problem statement.

    // Functional Utility: Calculates the sum of the three opinions.
    auto sum = x + y + z; // @var sum: The sum of the three opinions.

    /**
     * @brief Checks if at least two out of three opinions are positive (implying agreement).
     * @details A sum of 2 or more indicates that at least two individuals agree to implement the problem.
     * @block_logic Increments `count` if the sum of opinions is 2 or more.
     * @invariant `count` stores the number of problems where at least two opinions are positive.
     */
    if (sum >= 2)
      count++; // Increments the counter if the problem meets the implementation criteria.
  }

  // Functional Utility: Prints the total count of problems that will be implemented to standard output.
  std::cout << count << std::endl;

  return 0; // Indicates successful execution.
}

