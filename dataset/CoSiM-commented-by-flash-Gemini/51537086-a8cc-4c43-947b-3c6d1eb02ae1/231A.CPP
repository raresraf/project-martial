/**
 * @file 231A.CPP
 * @brief Solution to a problem involving team problem-solving capability.
 *        This program reads the number of problems and for each problem,
 *        three votes from participants. It counts how many problems have
 *        at least two 'yes' votes (represented by '1').
 *
 * Algorithm: Iterative counting. For each problem, it sums up the 'yes' votes.
 *            If the sum is 2 or more, it increments a global counter for solvable problems.
 * Time Complexity: O(N * P), where N is the number of problems and P is the
 *                  number of participants per problem (constant 3 in this case).
 *                  So, effectively O(N).
 * Space Complexity: O(1) as it uses a fixed amount of memory regardless of input size.
 */

#include <iostream> // Required for input/output operations (cin, cout)
using namespace std;

int main() {
  /**
   * Functional Utility: Manages the input and processing for a series of problems
   *                     to determine the total count of solvable problems based on
   *                     team consensus.
   */
  int n; // Stores the total number of problems to evaluate.
  cin >> n; // Reads the number of problems from standard input.

  int count = 0; // Initializes a counter for problems that can be solved.

  /**
   * Block Logic: Iterates through each problem to determine if it's solvable.
   * Pre-condition: `n` holds the number of problems. `count` is initialized to 0.
   * Invariant: `n` decrements with each iteration; the loop continues until all
   *            problems are evaluated. `count` reflects the number of solvable
   *            problems encountered so far.
   */
  while (n--) {
    int cnt = 0, temp; // `cnt` counts 'yes' votes for the current problem; `temp` stores each vote.

    /**
     * Block Logic: Reads three votes for the current problem and counts 'yes' votes.
     * Pre-condition: `cnt` is 0.
     * Invariant: After each iteration, `cnt` is updated if `temp` is '1'.
     *            The loop runs exactly 3 times for each problem.
     */
    for (int i = 0; i < 3; i++) {
      cin >> temp; // Reads a participant's vote.
      if (temp == 1)
        cnt++; // If the vote is '1' (yes), increment the 'yes' vote counter.
    }

    // Condition Check: If at least two participants voted 'yes'.
    if (cnt >= 2)
      count++; // Increment the total count of solvable problems.
  }

  // Output: Prints the final total count of problems that can be solved.
  cout << count << endl;

  return 0; // Indicates successful program termination.
}
