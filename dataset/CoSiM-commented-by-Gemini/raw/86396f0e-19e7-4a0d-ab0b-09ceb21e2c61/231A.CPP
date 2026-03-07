/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from competitive programming.
 *
 * @details
 * The program calculates the number of problems a three-person team will implement.
 * The condition for implementation is that at least two of the three members are
 * confident in the solution. This implementation is space-efficient, processing
 * each problem's data as it is read.
 *
 * Algorithm:
 * 1. Read the total number of problems, n.
 * 2. Initialize a counter for solvable problems to zero.
 * 3. Loop `n` times using a `while(n--)` construct.
 *    a. For each problem, read the three confidence levels (0 or 1).
 *    b. If the sum of the confidence levels is 2 or greater, increment the counter.
 * 4. Print the final counter value.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1), as input is processed on the fly without being stored.
 */
#include <iostream>

int main() {
  // Disables synchronization with C-style I/O for faster input/output.
  std::ios::sync_with_stdio(false);
  int n;
  std::cin >> n;
  int a, b, c, count = 0;
  /**
   * @brief Loop that iterates 'n' times to process each problem.
   * The `n--` is a common C++ idiom for a countdown loop in competitive programming.
   */
  while (n--) {
    // Read the confidence levels for the current problem.
    std::cin >> a >> b >> c;
    // Check if at least two members are confident.
    if ((a + b + c) >= 2) {
      count++;
    }
  }
  // Print the total number of problems the team will solve.
  std::cout << count << std::endl;
}