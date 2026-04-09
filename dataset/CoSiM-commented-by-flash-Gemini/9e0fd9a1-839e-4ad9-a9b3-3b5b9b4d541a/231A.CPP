/**
 * @file 231A.CPP
 * @brief This program simulates a team's problem-solving decision process in a programming contest.
 * For each problem, three team members express their opinion (1 if they are sure about the solution,
 * 0 otherwise). A problem is considered solvable if at least two team members are sure about the solution.
 * The program counts the total number of problems that the team will attempt to solve.
 *
 * Algorithm: Iterative decision evaluation.
 * 1. Read the total number of problems `n`.
 * 2. For each of `n` problems:
 *    a. Read three binary integers (0 or 1) representing the opinions of the three team members.
 *    b. Sum these three opinions.
 *    c. If the sum is 2 or greater, increment a counter for solvable problems.
 * 3. Print the final count of solvable problems.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * The outer loop runs `N` times, and the inner operations (reading 3 integers, summing, and comparing)
 * take constant time. Thus, the total time complexity is linear with respect to the number of problems.
 *
 * Space Complexity: O(1).
 * Only a few integer variables and a fixed-size array of 3 integers are used,
 * regardless of the number of problems `n`.
 */

#include <iostream> // Required for standard input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function of the program.
 * Reads problem data, applies the team decision rule, and outputs the total count
 * of problems the team will solve.
 * @pre Standard input consists of:
 *      - An integer `n` (1 <= n <= 1000), representing the number of problems.
 *      - Followed by `n` lines, each containing three space-separated integers
 *        (each being 0 or 1) representing the opinions of the three team members
 *        for a particular problem.
 * @post A single integer representing the total count of problems that are
 *       considered solvable by the team is printed to standard output, followed by a newline.
 * @return 0 upon successful program execution, indicating no errors.
 */
int main() {
  // Functional Utility: `arr` temporarily stores the opinions of the three team members for a single problem.
  // `i`, `j` are loop counters. `n` stores the total number of problems.
  // `flag` counts the number of problems where at least two team members agree.
  int arr[3], i, j, n, flag = 0;

  // Functional Utility: Reads the total number of problems `n` from standard input.
  cin >> n;

  // Block Logic: Validates the input `n` against typical competitive programming constraints.
  // Ensures `n` is between 1 and 1000 (inclusive), preventing processing for out-of-range values.
  if (1 <= n && n <= 1000) {
    /**
     * Block Logic: Outer loop that iterates `n` times, once for each problem to be evaluated.
     * Invariant: At the start of each iteration `i`, `flag` correctly reflects the number of
     *            solvable problems among the first `i` problems processed.
     * Pre-condition: `n` has been read and validated as within the acceptable range.
     */
    for (i = 0; i < n; i++) {
      /**
       * Block Logic: Inner loop that reads the three binary opinions for the current problem.
       * These opinions are stored in the `arr` array.
       * Invariant: After each iteration `j`, `arr[j]` contains the opinion of the `(j+1)`-th
       *            team member for the current problem.
       * Pre-condition: Standard input provides three integers for the current problem.
       */
      for (j = 0; j < 3; j++) {
        cin >> arr[j];
      }
      /**
       * Block Logic: Evaluates the team's decision rule for the current problem.
       * If the sum of the three opinions is 2 or more, it means at least two members agree,
       * and the problem is considered solvable.
       * Pre-condition: `arr` contains the three binary opinions for the current problem.
       */
      if (arr[0] + arr[1] + arr[2] >= 2)
        flag++; // Functional Utility: Increments the `flag` counter if the problem is solvable.
    }
    // Functional Utility: Prints the final count of problems that the team will solve, followed by a newline.
    cout << flag << endl;
  }
  return 0; // Functional Utility: Indicates successful program termination.
}
