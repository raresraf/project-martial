/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The problem is to count how many problems a team of three will solve.
 * The team decides to solve a problem only if at least two of the three members
 * are confident in the solution.
 *
 * @algorithm
 * The solution reads the total number of problems, `n`. It then iterates `n` times,
 * processing one problem in each iteration. For each problem, it reads the three
 * members' confidence levels (1 for sure, 0 for not). Instead of summing the
 * confidence levels, this implementation uses a series of logical ORs (`||`) to
 * check all possible pairs of members being sure (`p` and `v`, `p` and `t`, or `v` and `t`).
 * If any of these conditions are met, it means at least two members are sure, and a
 * counter for solvable problems is incremented. Finally, the total count is printed.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, due to the single loop
 * that runs N times.
 * Space Complexity: O(1), as it only uses a fixed number of integer variables.
 */
#include <iostream>
using namespace std;

int main() {
  /**
   * @variable n: The total number of problems to be evaluated.
   */
  int n;
  cin >> n;
  /**
   * @variable count: A counter for the number of problems the team will solve.
   */
  int count = 0;
  /**
   * @block
   * @brief This loop iterates through each of the `n` problems.
   * @invariant `count` holds the number of problems solved so far.
   */
  while (n--) {
    /**
     * @variable p, v, t: Represent the confidence of the three team members.
     * 1 indicates they are sure of the solution, 0 otherwise.
     */
    int p, v, t;
    cin >> p >> v >> t;

    // Pre-condition: Check if at least two members are sure about the solution.
    // This is done by checking all three possible pairs of confident members.
    if ((p == 1 && v == 1) || (p == 1 && t == 1) || (v == 1 && t == 1)) {
      count++; // Increment the count of solvable problems.
    }
  }
  // Output the final total.
  cout << count;
}
