/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - "Team".
 * @details This program calculates the number of problems a team of three will solve.
 *          The condition for solving a problem is that at least two members of the team
 *          are confident in the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for solved problems (`kq`) to zero.
 * 3. Loop `n` times, once for each problem:
 *    a. Read the confidence levels (1 for sure, 0 for not sure) of the three members.
 *    b. Count how many members are sure (`dem`).
 *    c. If the count of sure members is 2 or more, increment the solved problems counter.
 * 4. Print the total number of solved problems.
 *
 * Time Complexity: O(N), where N is the number of problems, as it processes each problem in constant time.
 * Space Complexity: O(1), as the memory usage is constant.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  // 'kq' (Vietnamese: "kết quả") stores the final count of problems to be solved.
  int kq = 0;
  cin >> n;

  /**
   * @brief Main loop to iterate through each of the `n` problems.
   */
  while (n--) {
    // 'dem' (Vietnamese: "đếm") counts the number of members sure about the current problem.
    int dem = 0;
    int a[3];
    /**
     * @brief Inner loop to read the confidence of the three team members.
     * For each problem, it reads three integers (0 or 1).
     */
    for (int i = 0; i < 3; i++) {
      cin >> a[i];
      // If a member is sure (input is 1), increment the confidence counter.
      if (a[i] == 1)
        dem++;
    }
    /**
     * @brief Check if the team will solve the problem.
     * The condition is that at least two members are confident.
     */
    if (dem >= 2)
      kq++;
  }
  // Output the total number of problems that will be solved.
  cout << kq;
  return 0;
}
