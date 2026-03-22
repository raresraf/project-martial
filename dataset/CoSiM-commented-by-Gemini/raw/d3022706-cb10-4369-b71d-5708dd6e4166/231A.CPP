/**
 * @file 231A.CPP
 * @brief An alternative solution for the "Team" problem from Codeforces (Problem 231A).
 * @details This program determines how many problems a team of three will solve,
 * given the condition that at least two members must be sure of the solution.
 * This specific implementation stores all inputs in a 2D array before
 * using a series of explicit pairwise comparisons to check the condition.
 *
 * @note This code uses a Variable Length Array (VLA) (`a[n][3]`), which is a
 * non-standard C++ extension supported by compilers like GCC and Clang, but
 * is not part of the C++ standard.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the number of problems, N.
 * 2. Declare a VLA of size N x 3 to store all inputs.
 * 3. Read all N*3 opinions into the 2D array.
 * 4. Iterate through each problem (row) in the array.
 * 5. For each problem, use a chain of `if-else if` statements to check if
 *    any two members agree (i.e., their opinions are both 1).
 * 6. If the condition is met, increment a counter and move to the next problem.
 * 7. Print the final counter value.
 *
 * @section complexity
 * - Time Complexity: O(N), for reading and processing N problems.
 * - Space Complexity: O(N), due to the VLA used to store all inputs.
 */
#include <iostream>
#include <string>
#define ll long long
#define ld long double

int main() {
  int n, output = 0;
  std::cin >> n;
  // Non-standard C++: a Variable Length Array (VLA) is declared here.
  int a[n][3];

  /**
   * @brief Reads all problem inputs into the 2D array.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      std::cin >> a[i][j];
    }
  }

  /**
   * @brief Processes each problem to count the number of solvable ones.
   * @invariant `output` holds the count of problems found to be solvable.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @brief This chain of if-else if statements is a verbose but correct
     * way to check if the sum of opinions is >= 2. It explicitly checks
     * all pairs of members for agreement.
     */
    if (a[i][0] == 1 && a[i][0] == a[i][1]) { // Checks members 1 and 2
      output++;
      continue;
    } else if (a[i][1] == 1 && a[i][1] == a[i][2]) { // Checks members 2 and 3
      output++;
      continue;
    } else if (a[i][0] == 1 && a[i][0] == a[i][2]) { // Checks members 1 and 3
      output++;
      continue;
    }
  }

  std::cout << output;

  return 0;
}