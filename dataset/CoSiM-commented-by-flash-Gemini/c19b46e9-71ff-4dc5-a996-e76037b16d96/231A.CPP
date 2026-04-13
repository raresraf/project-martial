/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem. It counts the number of problems
 * a team will solve, given their confidence levels. For each problem, three friends
 * state their confidence (1 if sure, 0 if not). A problem is solved if at least two
 * friends are sure (sum of confidence levels is 2 or more).
 * Algorithm: Iterative processing of problems, summing confidence levels, and conditional counting.
 * Time Complexity: O(N * C) where N is the number of problems and C is the fixed number of friends (3).
 * Space Complexity: O(C) for storing confidence levels in the vector for each problem.
 */
//#include <bits/stdc++.h> // Commented out in original, but typically provides common headers.
// using namespace std; // Commented out in original, replaced with std:: prefix.
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <vector>   // Required for using std::vector to store confidence levels.

int main() {
  // Disables synchronization between C-style I/O and C++ streams, and unties cin from cout.
  // This optimizes input/output operations for competitive programming.
  std::ios::sync_with_stdio(0);
  std::cin.tie(0);

  int n;          // Stores the total number of problems to be considered.
  int counter = 0;    // Counts the number of problems the team will solve.
  int vector_sum = 0; // Accumulates the sum of confidence levels for the current problem.
  std::vector<int> v; // A dynamic array to temporarily store the three confidence levels for a problem.
  // Reads the total number of problems 'n'.
  std::cin >> n;

  /**
   * Block Logic: Processes each of the 'n' problems.
   * For each problem, it reads the confidence levels of three friends, calculates their sum,
   * and determines if the problem is solved.
   * Invariant: 'counter' correctly reflects the number of solved problems up to the current iteration.
   */
  for (int j = 0; j < n; j++) {
    /**
     * Block Logic: Reads the three confidence levels for the current problem.
     * Each confidence level (0 or 1) is read into 'x' and added to the vector 'v'.
     */
    for (int i = 0; i < 3; i++) {
      int x; // Stores the confidence level of a single friend.
      std::cin >> x;
      v.push_back(x);
    }
    /**
     * Block Logic: Calculates the sum of confidence levels for the current problem.
     * Iterates through the vector 'v' to sum up the individual confidence levels.
     */
    for (int &it : v) { // Uses a range-based for loop for convenience.
      vector_sum += it;
    }
    /**
     * Conditional Logic: Checks if the problem is solved.
     * If the sum of confidence levels ('vector_sum') is 2 or more, the problem is considered solved.
     */
    if (vector_sum >= 2) {
      counter++; // Increments the counter for solved problems.
    }
    vector_sum = 0; // Resets the sum for the next problem.
    v.clear();      // Clears the vector for the next problem's confidence levels.
  }
  // Prints the total number of problems the team will solve.
  std::cout << counter;

  // Standard program exit.
  return 0;
}
