/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A).
 *   It counts the number of problems for which at least two out of three participants are sure about the solution.
 *   Problem description: https://codeforces.com/problemset/problem/231/A
 * * Algorithm: Iterative counting. It reads 'n' sets of three integers (0 or 1),
 *   counts the number of '1's in each set, and then increments a total counter
 *   if a set has at least two '1's.
 * Time Complexity: O(N), where N is the number of problems, as the input reading,
 *   counting, and final summation each take linear time with respect to N.
 * Space Complexity: O(N) for storing the 2D array `a` and the `count` array.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
#define pi 2.0 * acos(0.0) // Defines pi, common in competitive programming but not used here.
#define MAX 100005 // Defines a maximum size, likely for array bounds.
typedef long long ll; // Functional Utility: Alias for long long, used for potentially large integers.
using namespace std;

/**
 * @brief Main function to execute the problem counting logic.
 * Functional Utility: Reads inputs for multiple problems, processes them to count
 *                     sure participants, and outputs the total number of problems
 *                     meeting the criteria.
 */
int main() {
  // `a`: A 2D array to store the certainty votes (0 or 1) for each participant for each problem.
  //      Declared with a fixed size [2000][3], implying up to 2000 problems.
  // `count`: An array to store the sum of '1's (sure participants) for each problem.
  // `i, j, k`: Loop counters.
  // `n`: Stores the total number of problems.
  // `c`: Accumulates the total count of problems where at least two participants are sure.
  int a[2000][3], count[MAX], i, j, k, n, c;
  c = 0; // Initializes the problem counter to zero.
  // Block Logic: Reads the total number of problems, 'n'.
  cin >> n;
  /**
   * Block Logic: Initializes the `count` array to zeros.
   * Pre-condition: `n` is known.
   * Invariant: All elements of `count` from index 0 to `n-1` are set to 0.
   */
  for (i = 0; i < n; i++)
    count[i] = 0;
  /**
   * Block Logic: Reads the certainty votes for each participant for each problem.
   * Pre-condition: `n` is known.
   * Invariant: After these nested loops, `a[i][j]` contains 0 or 1, representing
   *            participant `j`'s certainty for problem `i`.
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  /**
   * Block Logic: Calculates the number of sure participants for each problem.
   * Pre-condition: The `a` array is populated with participant votes.
   * Invariant: After these nested loops, `count[i]` stores the total number of
   *            '1's (sure participants) for problem `i`.
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      if (a[i][j] == 1) // If a participant is sure (vote is 1).
        count[i]++;     // Increment the sure count for the current problem.
    }
  }
  /**
   * Block Logic: Determines the total number of problems where at least two participants are sure.
   * Pre-condition: `count` array contains the sum of sure participants for each problem.
   * Invariant: `c` accumulates the total count of qualifying problems.
   */
  for (i = 0; i < n; i++) {
    if (count[i] >= 2) { // If two or more participants are sure for problem `i`.
      c = c + 1;         // Increment the overall counter.
    }
  }
  // Block Logic: Prints the final total count of problems that meet the criteria.
  cout << c;
  return 0;
}
