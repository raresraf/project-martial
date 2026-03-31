/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve based on member confidence.
 *
 * This program is a solution to the "Team" problem. It reads an integer `n`,
 * which is the number of problems. For each problem, it reads the opinions of
 * three team members (1 for confident, 0 for not). If two or more members are
 * confident, the team decides to solve the problem. The program calculates and
 * prints the total number of problems the team will solve.
 *
 * @author f2d916d7-982e-45eb-a2bd-1bcf9692e427
 *
 * @algorithm Nested Loop Counting
 * @complexity
 * Time: O(N), as the outer loop runs N times and the inner loop runs a constant (3) times.
 * Space: O(1), since the memory used is constant regardless of the input size.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // `n`: number of problems. `a`: array to store 3 member opinions.
  // `c`: counts confident members for one problem. `s`: counts total solved problems.
  int n, a[3], c = 0, s = 0, i;
  cin >> n;

  /**
   * @brief Main loop to iterate through each of the `n` problems.
   * `n--` is used as the loop condition.
   */
  while (n--) {
    // Reset the confidence counter for the current problem.
    c = 0;
    /**
     * @brief Inner loop to read opinions of the 3 team members.
     */
    for (i = 0; i < 3; i++) {
      cin >> a[i];
      // If a member is confident, increment the counter for this problem.
      if (a[i] == 1) {
        c++;
      }
      // Debugging line `// cout<<n<<" "<<i<<endl;` is commented out.
    }
    /**
     * @brief Check if at least two members are confident.
     */
    if (c >= 2) {
      // If so, increment the total solved problems counter.
      s++;
    }
    // Debugging line `// cout<<c<<endl;` is commented out.
  }
  // Print the final count of solved problems.
  cout << s << endl;
}