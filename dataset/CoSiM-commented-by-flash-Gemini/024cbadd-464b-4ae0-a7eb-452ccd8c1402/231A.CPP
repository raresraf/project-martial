/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem (Codeforces 231A).
 * This program determines the number of problems a team can implement.
 * Each problem is proposed by three friends, and it is implemented only if
 * at least two out of the three friends are confident in its solution.
 * Algorithm: Iterative counting. The program reads the confidence levels (1 for sure, 0 for not sure)
 * of three friends for 'n' problems. For each problem, it sums the confidence levels,
 * and if the sum is 2 or more, it increments a counter for implemented problems.
 * Time Complexity: O(N), where N is the number of problems, as the program iterates through each problem once
 * to read input and once again to process it.
 * Space Complexity: O(N), for storing the 2D array `a[N][3]` which holds the confidence levels for each problem.
 */
#include <iostream>
using namespace std;
int main() {
  int n, count = 0; // Inline: 'n' for number of problems, 'count' for implemented problems.
  int a[1000][3]; // Inline: 2D array to store confidence levels (1=sure, 0=not sure) for up to 1000 problems by 3 friends.
  // Functional Utility: Reads the total number of problems 'n'.
  cin >> n;
  // Block Logic: Reads the confidence levels for each problem.
  // Precondition: 'n' is a positive integer representing the number of problems.
  // Invariant: After this loop, 'a[i][j]' will contain the confidence level of friend 'j' for problem 'i'.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  // Block Logic: Processes each problem to determine if it will be implemented.
  // Invariant: 'count' accurately reflects the number of problems that will be implemented.
  for (int i = 0; i < n; i++) {
    int flag = 0; // Inline: 'flag' counts friends who are sure about the current problem's solution.
    // Conditional Logic: Checks if the first friend is sure about the solution.
    if (a[i][0] == 1)
      flag++;
    // Conditional Logic: Checks if the second friend is sure about the solution.
    if (a[i][1] == 1)
      flag++;
    // Conditional Logic: Checks if the third friend is sure about the solution.
    if (a[i][2] == 1)
      flag++;
    // Conditional Logic: If at least two friends are sure (flag >= 2), the problem is implemented.
    if (flag >= 2)
      count++;
  }
  // Functional Utility: Prints the total number of problems that will be implemented.
  cout << count;
  return 0;
}
