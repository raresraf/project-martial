/**
 * @file 231A.CPP
 * @brief A space-efficient solution to the "Team" problem from Codeforces.
 * @details This program counts the number of problems a three-person team will
 * implement. The condition for implementation is that at least two of the three
 * members must be confident in the solution.
 *
 * @algorithm
 * This solution uses a nested loop and processes the input on the fly, which
 * is very efficient in terms of memory.
 * 1. Read the number of problems, `n`.
 * 2. An outer loop iterates `n` times, once for each problem.
 * 3. For each problem, an inner loop runs 3 times to read the opinions of the
 *    three members. The opinions (0 or 1) are summed up in a temporary counter.
 * 4. After the inner loop, if the sum of opinions for the current problem is 2
 *    or greater, a main result counter is incremented.
 * 5. After all problems are processed, the final result is printed.
 *
 * @complexity
 * Time Complexity: O(N), as the program reads 3*N integers in total.
 * Space Complexity: O(1), as the inputs are processed without being stored in
 * an array.
 */
#include <bits/stdc++.h>
using namespace std;

// Unused type definitions, likely part of a personal template.
typedef long long ll;
typedef pair<int, int> ii;

int main() {
  // Standard competitive programming optimizations for fast I/O.
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);

  int n; // The total number of problems.
  int x; // A temporary variable to hold a single member's opinion (0 or 1).
  int solvable_problems_count = 0; // The final result counter.
  cin >> n;

  /**
   * @block
   * @brief Outer loop to iterate through each of the `n` problems.
   */
  for (int i = 0; i < n; i++) {
    // Counter for the number of members sure about the *current* problem.
    int current_problem_votes = 0;
    /**
     * @block
     * @brief Inner loop to process the 3 opinions for the current problem.
     */
    for (int j = 0; j < 3; j++) {
      cin >> x;
      current_problem_votes += x;
    }

    // If two or more members are sure, the team will solve the problem.
    if (current_problem_votes >= 2)
      solvable_problems_count++;
  }

  // Print the total number of problems that will be solved.
  cout << solvable_problems_count << endl;
  return 0;
}