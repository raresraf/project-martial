/**
 * @file 231A.CPP
 * @brief Solution for competitive programming problem: "Team".
 *
 * @details This program determines the number of problems that can be implemented.
 *          For each problem, three participants provide votes (0 or 1). A problem
 *          is considered implementable if at least two participants vote positively (1).
 *          Algorithm: Iterative processing of problem votes using a helper function
 *                     to check the implementability condition.
 *          Time Complexity: O(N), where N is the number of problems, as each problem's
 *                           votes are processed in constant time.
 *          Domain: Competitive Programming.
 */
#include <iostream>

using namespace std;
bool can_be_implemented(int *);

int main(void) {
  // Declare 'n' for the total number of problems.
  int n;
  // Read the total number of problems from standard input.
  cin >> n;

  // Declare a 2D array to store the votes for all 'n' problems.
  // Each row represents a problem, and columns store votes from 3 participants.
  int data[n][3];

  // Block Logic: Take input for all 'n' problems.
  // Invariant: The loop iterates 'n' times, reading 3 votes for each problem.
  for (int i = 0; i < n; i++)
    cin >> data[i][0] >> data[i][1] >> data[i][2]; // Read votes for the current problem.

  // Initialize a counter for problems that can be implemented.
  int implement_solution_count = 0;

  // Block Logic: Process each problem to determine if it can be implemented.
  // Invariant: The loop iterates 'n' times, checking each problem's implementability.
  for (int i = 0; i < n; i++) {
    // Call the helper function to check if the current problem's votes allow implementation.
    if (can_be_implemented(data[i]))
      implement_solution_count++; // Increment count if the problem is implementable.
  }

  // Output the total count of implementable problems.
  cout << implement_solution_count << endl;

  return 0; // Indicate successful program termination.
}

/**
 * @brief Checks if a problem can be implemented based on participant votes.
 *
 * @details This helper function takes an array of three integer votes (0 or 1)
 *          and determines if at least two of them are positive (1).
 *
 * @param arr A pointer to an integer array containing three votes (0 or 1).
 * @return `true` if two or more votes are 1, `false` otherwise.
 */
bool can_be_implemented(int *arr) {
  int one_count = 0; // Counter for positive votes (1s).
  // Block Logic: Iterate through the three votes and count the positive ones.
  // Invariant: 'one_count' accurately reflects the number of '1's in 'arr'.
  for (int i = 0; i < 3; i++)
    if (arr[i] == 1) // If the current vote is 1.
      one_count++;   // Increment the positive vote counter.

  // Block Logic: Determine if the problem is implementable.
  // Precondition: At least two positive votes are required for implementation.
  if (one_count >= 2)
    return true; // Problem can be implemented.
  else
    return false; // Problem cannot be implemented.
}