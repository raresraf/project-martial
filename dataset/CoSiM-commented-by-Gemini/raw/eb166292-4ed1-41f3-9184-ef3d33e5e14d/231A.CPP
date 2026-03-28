/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 *
 * This program counts the number of problems a team of three will implement,
 * based on the rule that at least two members must be sure about the solution.
 *
 * @algorithm The program iterates through the `n` problems. For each problem,
 * it reads the three members' opinions (1 for sure, 0 for not sure), calculates
 * their sum, and if the sum is greater than 1 (i.e., 2 or 3), it increments a
 * counter for solvable problems. The final count is then printed.
 *
 * @time_complexity O(N), where N is the number of problems.
 * @space_complexity O(1).
 */
//#include <bits/stdc++>
#include <iostream>

int main() {
  int total = 0;
  int a, b, c;
  a = 0;
  b = 0;
  c = 0;
  int j = 0; // Temporary variable to hold the sum of opinions for one problem.
  int total_to_use = 0; // Counter for problems the team will solve.
  std::cin >> total;
  /**
   * @block_logic This loop iterates through each of the `total` problems.
   */
  for (int i = 0; i < total; i++) {
    std::cin >> a;
    std::cin >> b;
    std::cin >> c;
    j += a + b + c; // Sum the opinions.
    // If the sum is 2 or more, the team solves the problem.
    if (j > 1)
      total_to_use++;
    // The following variable resets are redundant for the next loop iteration.
    j = 0;
    a = 0;
    b = 0;
    c = 0;
  }
  std::cout << total_to_use;
  return (0);
}
