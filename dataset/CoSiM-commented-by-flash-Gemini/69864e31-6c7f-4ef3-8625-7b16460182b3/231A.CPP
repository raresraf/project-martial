/**
 * @69864e31-6c7f-4ef3-8625-7b16460182b3/231A.CPP
 * @brief Problem: Team. Determines the number of problems a team can implement.
 *        A team of three friends solves a problem if at least two of them are sure about the solution.
 *        Given 'n' problems, and for each problem, the certainty of three friends (1 if sure, 0 if not),
 *        the task is to count how many problems the team will implement.
 * Algorithm: Iterative summation and thresholding.
 *            For each problem, sum the certainty levels of the three friends.
 *            If the sum is greater than 1 (meaning at least two friends are sure), increment a counter.
 * Time Complexity: O(N) where N is the number of problems.
 *                  The outer loop runs N times, and the inner loop runs a constant 3 times.
 * Space Complexity: O(N) for storing the certainty matrix.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  // Block Logic: Reads the total number of problems from standard input.
  cin >> n;
  
  // Functional Utility: Declares a 2D array to store the certainty of three friends for 'n' problems.
  // Invariant: 'a[i][j]' stores 1 if friend 'j' is sure about problem 'i', otherwise 0.
  int a[n][3], sum, count = 0;

  // Block Logic: Populates the certainty matrix 'a' by reading input for each problem and each friend.
  // Pre-condition: 'n' has been read and is a positive integer.
  // Invariant: After each inner loop iteration, 'a[i]' contains the certainty levels for problem 'i'.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }

  // Block Logic: Iterates through each problem to determine if the team will implement it.
  // Pre-condition: The 'a' matrix is fully populated with certainty levels.
  // Invariant: 'count' accurately reflects the number of problems implemented up to problem 'i'.
  for (int i = 0; i < n; i++) {
    sum = 0;
    // Block Logic: Calculates the sum of certainty levels for the current problem.
    // Invariant: 'sum' accumulates the certainty scores for problem 'i'.
    for (int j = 0; j < 3; j++) {
      sum += a[i][j];
    }
    // Block Logic: Checks if at least two friends are sure about the current problem's solution.
    // Pre-condition: 'sum' contains the total certainty score for problem 'i'.
    // Post-condition: 'count' is incremented if 'sum' is greater than 1.
    if (sum > 1)
      count += 1;
  }
  // Functional Utility: Prints the total number of problems the team will implement.
  cout << count;
  // your code goes here

  return 0;
}
