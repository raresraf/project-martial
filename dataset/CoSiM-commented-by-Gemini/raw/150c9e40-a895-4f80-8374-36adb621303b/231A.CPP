/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem.
 * @details The program counts the number of problems a team of three will solve.
 * The team decides to solve a problem if at least two of its three members are confident
 * about the solution. The input consists of `n` problems, and for each problem,
 * three boolean values (0 or 1) representing the confidence of each member.
 *
 * @section algorithm Algorithm
 * 1. Read the number of problems, `n`.
 * 2. A 2D array `data[n][3]` is used to store the inputs.
 * 3. The decisions for all `n` problems are read into this array.
 * 4. A counter for the number of problems to be solved is initialized to 0.
 * 5. The program iterates through each problem's decisions:
 *    a. It sums the three integer values (0s and 1s) for the current problem.
 *    b. If the sum is 2 or greater, it signifies that at least two members are sure,
 *       so the problem counter is incremented.
 * 6. Finally, the total count of solvable problems is printed.
 *
 * @note This code uses a Variable Length Array (VLA) `data[n][3]`, which is not
 * part of the standard C++ language but is supported as an extension by some compilers like g++.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of problems, as it requires reading N lines and then iterating through them once.
 * - Space Complexity: O(N), to store the decisions for all `n` problems in the 2D array.
 */
#include <iostream>
using namespace std;

int main() {
  int n, count = 0;
  cin >> n;
  // A Variable Length Array (VLA) is used here. This is a C99 feature and a non-standard extension in C++.
  int data[n][3];

  // Read the confidence levels for each problem from each of the three members.
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 3; j++)
      cin >> data[i][j];

  /**
   * @brief Count the number of problems where at least two members are sure.
   * This is done by summing the confidence levels for each problem and checking
   * if the sum is 2 or more.
   */
  for (int i = 0; i < n; i++)
    if (data[i][0] + data[i][1] + data[i][2] >= 2)
      count++;

  // Output the total number of problems the team will solve.
  cout << count << endl;

  return 0;
}