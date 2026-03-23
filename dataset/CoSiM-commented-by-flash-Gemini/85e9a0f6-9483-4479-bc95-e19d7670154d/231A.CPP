/**
 * @file 231A.CPP
 * @brief This competitive programming solution addresses a problem where three friends
 *        decide to implement a problem if at least two of them are sure about the solution.
 *        Given a list of problems and each friend's confidence (1 for sure, 0 for not sure),
 *        the task is to count how many problems they will implement.
 * Algorithm: Iterative Counting.
 *            1. Read the number of problems `n`.
 *            2. For each problem, read the confidence levels of the three friends.
 *            3. For each problem, count how many friends are sure about the solution.
 *            4. If the count of sure friends is 2 or more, increment a global counter for implemented problems.
 *            5. Finally, print the total count of implemented problems.
 * Time Complexity: O(N), where N is the number of problems. The program iterates through
 *                  each problem and its three confidence values once.
 * Space Complexity: O(N) for storing the confidence values of all problems in a 2D array.
 */
#include <bits/stdc++.h> // Includes common standard library headers, including iostream.
using namespace std;

/**
 * @brief Main function of the program. Reads problem data, counts implementable problems, and prints the result.
 * Functional Utility: Serves as the entry point for execution, solving the problem of counting
 *                     problems with sufficient team confidence.
 */
int main() {
  int n; // Declares an integer to store the number of problems.
  cin >> n; // Reads the number of problems.
  // Declares a 2D array `a` to store the confidence levels for each problem.
  // Each row represents a problem, and the three columns represent the three friends.
  int a[n][3];
  
  // Block Logic: Reads the confidence levels for all `n` problems from standard input.
  // Invariant: After these nested loops, the `a` array is populated with the input data.
  for (int i = 0; i < n; i++) { // Outer loop iterates through each problem.
    for (int j = 0; j < 3; j++) { // Inner loop iterates through the three friends for the current problem.
      cin >> a[i][j]; // Reads the confidence level (0 or 1) for a friend on a problem.
    }
  }
  
  int count = 0; // Initializes a counter for problems that will be implemented.

  // Block Logic: Iterates through each problem to determine if it will be implemented.
  // Invariant: `count` accurately reflects the number of implementable problems processed so far.
  for (int i = 0; i < n; i++) {
    int count1 = 0; // Initializes a counter for friends who are sure about the current problem.
    // Inner Block Logic: Iterates through the three friends for the current problem.
    // Invariant: `count1` correctly sums the confidence levels for the current problem.
    for (int j = 0; j < 3; j++) {
      // If a friend is sure (value is 1), increment `count1`.
      if (a[i][j] == 1) {
        count1++;
      }
    }
    // Block Logic: Checks if at least two friends are sure about the current problem.
    if (count1 >= 2) {
      count++; // If yes, increment the total count of implementable problems.
    }
  }
  cout << count << endl; // Prints the final total count of problems that will be implemented.
}