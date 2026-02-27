/**
 * @file 231A.CPP
 * @brief Solution for competitive programming problem: "Team".
 *
 * @details This program counts the number of problems that can be implemented.
 *          For each problem, three participants cast votes (0 or 1). A problem
 *          is considered implementable if at least two participants vote positively (1).
 *          Algorithm: Direct simulation and counting based on conditional logic.
 *          Time Complexity: O(N), where N is the number of problems, as each problem's
 *                           votes are processed in constant time.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>
using namespace std;
main() {
  // Declare 'n' for the total number of problems.
  int n;
  // Read the total number of problems from standard input.
  cin >> n;
  // Declare 'c' as a counter for implementable problems, initialized to 0.
  int c = 0;
  
  // Block Logic: Iterate through each problem.
  // Invariant: The loop runs 'n' times, processing one problem per iteration.
  while (n--) {
    // Declare an array 'a' to store the votes of three participants for the current problem.
    int a[3];
    // Declare 'count' to store the number of positive votes (1s) for the current problem.
    int count = 0;
    // Block Logic: Read votes and count positive ones for the current problem.
    // Invariant: Iterates exactly 3 times for the three participants' votes.
    for (int i = 0; i < 3; i++) {
      cin >> a[i]; // Read a participant's vote.
      if (a[i] == 1) // If the vote is positive.
        count++;    // Increment the positive vote counter.
    }
    // Block Logic: Check if the current problem is implementable.
    // Precondition: A problem is implementable if at least two participants voted positively.
    if (count >= 2)
      c++; // Increment the total count of implementable problems.
  }
  // Output the total number of implementable problems.
  cout << c;
}
