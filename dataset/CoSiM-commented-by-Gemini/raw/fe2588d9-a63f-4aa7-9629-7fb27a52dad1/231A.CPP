/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem on platforms like Codeforces.
 * @details This program calculates the number of problems a team of three will implement.
 * The team decides to implement a problem only if at least two of the three members
 * are confident about the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, n.
 * 2. Initialize a 'counter' for implementable problems to 0.
 * 3. Loop n times, for each problem:
 *    a. Read the confidence levels of the three members (1 for sure, 0 for not sure).
 *    b. Sum the three confidence levels.
 *    c. If the sum is 2 or greater, it means at least two members are sure, so increment the counter.
 * 4. Print the final count of implementable problems.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream>

int main() {
  int n, a, b, c, counter = 0;
  std::cin >> n;
  /**
   * @brief Loop through each problem to check team's confidence.
   * Pre-condition: 'n' is the number of problems. 'counter' is 0.
   * Invariant: After each iteration, 'counter' holds the number of problems
   * that will be implemented so far.
   */
  for (int i = 0; i < n; i++) {
    std::cin >> a;
    std::cin >> b;
    std::cin >> c;

    // A problem is implemented if at least two members are sure.
    // The sum of confidence levels (0 or 1) directly gives the count of sure members.
    if (a + b + c >= 2) {
      counter += 1;
    }
  }
  // Output the total number of problems the team will implement.
  std::cout << counter << std::endl;
}