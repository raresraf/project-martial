/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem.
 *
 * This program determines the number of problems a team of three friends
 * will implement in a programming contest. A problem is implemented if
 * at least two of the three friends are sure about its solution.
 *
 * Algorithm: Iterative counting based on a sum threshold.
 * Time Complexity: O(N * M) where N is the number of problems and M is the
 *                  number of friends (fixed at 3). So, effectively O(N).
 * Space Complexity: O(N * M) to store the confidence levels in a 2D array.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to calculate the number of problems the team will implement.
 *
 * Functional Utility: Reads the number of problems and the confidence levels
 * for each friend per problem, then counts how many problems meet the
 * implementation criteria.
 */
int main() {
  int arr[1000][3]; // 2D array to store confidence levels for up to 1000 problems, 3 friends.
  int n;            // Number of problems.
  cin >> n;         // Read the total number of problems.

  /**
   * Block Logic: Reads the confidence levels for each friend for each problem.
   * Stores these 0/1 values in a 2D array.
   * Pre-condition: 'n' (number of problems) has been read.
   * Invariant: After processing each row (problem), `arr[i]` contains the confidence levels for problem `i`.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= 2; j++) {
      cin >> arr[i][j]; // Read confidence level (0 or 1) for friend 'j' on problem 'i'.
    }
  }

  int count = 0;   // Temporary counter for 'sure' friends per problem.
  int result = 0;  // Final count of problems to be implemented.

  /**
   * Block Logic: Iterates through each problem to determine if it will be implemented.
   * For each problem, it sums the confidence levels of the three friends.
   * If the sum is 2 or more, `result` is incremented.
   * Pre-condition: All problem confidence levels are stored in `arr`.
   * Invariant: `result` accurately reflects the number of implementable problems encountered so far.
   */
  for (int i = 0; i < n; i++) {
    count = 0; // Reset 'sure' friends counter for the current problem.
    for (int j = 0; j <= 2; j++) {
      if (arr[i][j] == 1) // If a friend is sure about the solution.
        count++;          // Increment the 'sure' friends counter.
      if (count == 2) { // If at least two friends are sure.
        result++;       // Increment the total count of implementable problems.
        break;          // No need to check the third friend if two are already sure.
      }
    }
  }
  cout << result; // Output the total number of problems to be implemented.
  return 0;
}
