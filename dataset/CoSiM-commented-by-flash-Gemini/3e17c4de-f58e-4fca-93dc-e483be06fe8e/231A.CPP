/**
 * @file 231A.CPP
 * @brief Solution for competitive programming problem: "Team".
 *
 * @details This program counts the number of problems that can be implemented.
 *          For each problem, three participants provide votes (0 or 1). A problem
 *          is considered implementable if at least two participants vote positively (1).
 *          Algorithm: Iterative processing of problem votes using a series of
 *                     conditional checks.
 *          Time Complexity: O(N), where N is the number of problems, as each problem's
 *                           votes are processed in constant time.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // Declare 'n' for the total number of problems, 'problem' to count implementable problems.
  int n, problem = 0;
  // Read the total number of problems from standard input.
  cin >> n;
  // Declare an array 'arr' to store the votes of three participants for a single problem.
  int arr[3];

  // Block Logic: Iterate 'n' times, processing each problem.
  // Invariant: 'problem' accumulates the count of implementable problems.
  while (n > 0) {
    // Block Logic: Read the three votes for the current problem.
    // Invariant: 'arr' holds the votes (0 or 1) for the current problem.
    for (int i = 0; i < 3; i++)
      cin >> arr[i]; // Read each participant's vote.

    // Block Logic: Check if at least two participants voted positively.
    // This uses a series of explicit checks for pairs and triples of positive votes.
    // While logically equivalent to `(arr[0] + arr[1] + arr[2]) >= 2`, this structure
    // explicitly checks combinations. The last condition `arr[0] == 1 && arr[1] == 1 && arr[2] == 1`
    // is redundant as it's covered by previous conditions.
    if (arr[0] == 1 && arr[1] == 1)
      problem++;
    else if (arr[1] == 1 && arr[2] == 1)
      problem++;
    else if (arr[0] == 1 && arr[2] == 1)
      problem++;
    else if (arr[0] == 1 && arr[1] == 1 && arr[2] == 1) // Redundant condition, already covered by the first.
      problem++;
    
    n--; // Decrement the problem counter.
  }
  // Output the total number of implementable problems.
  cout << problem;
}
