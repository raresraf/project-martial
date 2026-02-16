/**
 * @file 231A.CPP
 * @brief Solution for Codeforces problem 231A, "Team".
 *
 * @description
 * This program calculates the number of problems a team of three will solve.
 * For each problem, the input indicates whether each of the three members is
 * sure about the solution (1 for sure, 0 for not sure). The team will only
 * solve a problem if at least two members are sure. The program counts how
 * many problems meet this criterion.
 *
 * @algorithm
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for solvable problems, `r`, to 0.
 * 3. Loop `n` times to process each problem.
 *    a. Read the three members' opinions (0 or 1).
 *    b. Calculate the sum of the three opinions.
 *    c. If the sum is 2 or greater, it means at least two members are sure,
 *       so increment the counter `r`.
 * 4. After the loop, print the final count `r`.
 *
 * @time_complexity O(N), where N is the number of problems.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  // n: the total number of problems to consider.
  int n;
  cin >> n;

  // r: the counter for the number of problems that will be implemented.
  int r = 0;
  // Block Logic: Iterate through each of the 'n' problems.
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;

    // Block Logic: Check if the team will solve the current problem.
    // The sum of the inputs (0s and 1s) equals the number of members
    // who are sure about the solution.
    if (a + b + c >= 2) {
      r++;
    }
  }

  cout << r << endl;

  return 0;
}
