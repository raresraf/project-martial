/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details This program calculates the number of problems a three-person team will
 * implement. The team decides to solve a problem if at least two of the three
 * members are confident in the solution.
 *
 * @algorithm The program iterates through `n` problems. For each problem, it sums
 * the number of members who are sure about the solution (represented by '1').
 * If this sum is 2 or greater, a counter for problems to be implemented is
 * incremented. After checking all `n` problems, the total count is printed.
 *
 * @time_complexity O(N), where N is the number of problems, because the nested
 * loops result in 3*N iterations, which is linear in N.
 * @space_complexity O(1), as a fixed amount of memory is used regardless of the
 * number of problems.
 */
#include <iostream>

using namespace std;

int main() {

  // n: The total number of problems.
  // a: The opinion of a single team member (1 for sure, 0 for not).
  int n, a;
  // k: A temporary counter for 'sure' votes for the current problem.
  // count: The final count of problems to be implemented.
  int k, count = 0;

  cin >> n;

  /**
   * Block Logic: Iterate through each of the 'n' problems.
   */
  for (int i = 0; i < n; ++i) {
    k = 0; // Reset 'sure' vote count for the new problem.

    /**
     * Block Logic: Tally the votes for the current problem.
     * This loop runs 3 times, one for each team member.
     */
    for (int j = 0; j < 3; ++j) {
      cin >> a;
      if (a == 1) {
        k++;
      }
    }
    // Pre-condition: 'k' holds the number of members sure about the current problem.
    // If two or more members are sure, the team will implement it.
    if (k >= 2) {
      count++;
    }
  }

  cout << count << endl;

  return 0;
}