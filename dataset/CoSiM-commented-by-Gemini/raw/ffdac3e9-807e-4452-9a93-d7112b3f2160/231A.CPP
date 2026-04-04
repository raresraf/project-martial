/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem on Codeforces.
 *
 * @details This program determines how many problems a team of three will implement.
 * The condition for implementation is that at least two of the three members are
 * sure about the solution (represented by a '1').
 *
 * Algorithm:
 * 1. Read the total number of problems, n.
 * 2. Initialize a final counter `counter` to 0.
 * 3. Loop 'n' times for each problem:
 *    a. Initialize a temporary counter `temp` for the current problem to 0.
 *    b. Loop 3 times to read the opinions of the three members.
 *    c. For each member, if their input is 1, increment `temp`.
 *    d. After reading the three opinions, if `temp` (the number of sure members)
 *       is 2 or more, increment the final `counter`.
 * 4. Print the `counter`.
 *
 * Time Complexity: O(N), as the loops run a total of N * 3 times.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;

  int counter = 0;
  /**
   * @brief Main loop iterating through each of the 'n' problems.
   */
  while (n--) {
    int temp = 0;
    /**
     * @brief Inner loop to read three opinions and count the '1's.
     * `temp` will store the number of members sure about the current problem.
     */
    for (int i = 0; i < 3; i++) {

      int a;
      cin >> a;
      if (a == 1)
        temp++;
    }
    // If two or more members are sure, the team will solve the problem.
    if (temp >= 2)
      counter++;
  }
  cout << counter << "\n";
  return 0;
}
