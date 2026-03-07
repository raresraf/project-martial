/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem by first storing all inputs in a matrix.
 *
 * @details
 * This program determines how many problems a team will solve based on the confidence
 * of its three members. A problem is solved if at least two members are confident.
 * This implementation uses a 2D array to store the confidence levels for all
 * problems before processing them.
 *
 * Algorithm:
 * 1. Read the number of problems, n.
 * 2. Declare a variable-length array (VLA) of size n x 3 to store all inputs.
 *    Note: VLAs are a non-standard C++ extension.
 * 3. Read all confidence levels (0 or 1) and populate the n x 3 matrix.
 * 4. Initialize a counter for solvable problems.
 * 5. Iterate through each row of the matrix (each problem).
 *    a. For each problem, count how many members are confident (i.e., have a value of 1).
 *    b. If this count is 2 or greater, increment the solvable problems counter.
 * 6. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of problems. The program iterates through the N*3 inputs twice.
 * Space Complexity: O(N), due to the N x 3 matrix used to store all inputs.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  // Use a variable-length array (a non-standard C++ extension) to store all inputs.
  int a[n][3];
  
  /**
   * @brief Loop to read and store all confidence levels for all problems.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }

  int count = 0; // Counter for problems the team will solve.
  
  /**
   * @brief Loop to process the stored inputs and count solvable problems.
   * Iterates through each problem (row) in the matrix.
   */
  for (int i = 0; i < n; i++) {
    int count1 = 0; // Counter for confident members for the current problem.
    /**
     * @brief Inner loop to count confident members for problem 'i'.
     */
    for (int j = 0; j < 3; j++) {
      if (a[i][j] == 1) {
        count1++;
      }
    }
    // If two or more members are confident, the team solves the problem.
    if (count1 >= 2) {
      count++;
    }
  }
  cout << count << endl;
}
