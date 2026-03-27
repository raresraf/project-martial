/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (Problem 231A).
 * @details This program calculates the number of problems a team of three will
 * attempt. The condition for attempting a problem is that at least two of the
 * three members are confident about the solution.
 *
 * @section
 * Algorithm:
 * 1. Read the total number of problems, 'n'.
 * 2. For each of the 'n' problems, read the confidence levels (0 or 1) of the three members.
 * 3. Sum the confidence levels for each problem.
 * 4. If the sum is 2 or greater, increment a counter for the problems to be attempted.
 * 5. After checking all problems, print the final count.
 *
 * @section
 * Complexity Analysis:
 * - Time Complexity: O(n), where 'n' is the number of problems, due to the loops for reading input and processing.
 * - Space Complexity: O(n), for storing all confidence levels in a 2D array. This can be optimized to O(1) by processing each problem's data within the input loop.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  /**
   * Data Structure: A 2D array to store the confidence levels for n problems.
   * Note: Using a variable 'n' to declare an array size is a non-standard C++
   * feature (Variable Length Array - VLA), supported by compilers like GCC.
   * A more portable or memory-efficient approach for large 'n' would be to
   * process inputs line-by-line or use dynamic allocation (e.g., std::vector).
   */
  int sure[3][n];
  // Block Logic: Read the confidence levels for all n problems.
  for (int i = 0; i < n; i++)
    cin >> sure[0][i] >> sure[1][i] >> sure[2][i];

  int attempt = 0, c = 0;
  /**
   * Block Logic: Iterate through each problem to count how many will be solved.
   * Invariant: 'attempt' holds the count of problems to be solved up to the (i-1)-th problem.
   */
  for (int i = 0; i < n; i++) {
    // Sum the confidence levels for the current problem.
    c = sure[0][i] + sure[1][i] + sure[2][i];
    // Condition: If two or more members are sure, the team attempts the problem.
    if (c >= 2)
      ++attempt;
    c = 0; // Reset sum for the next iteration (though it's recalculated anyway).
  }

  // Output the total number of problems the team will attempt.
  cout << attempt << endl;

  return 0;
}
