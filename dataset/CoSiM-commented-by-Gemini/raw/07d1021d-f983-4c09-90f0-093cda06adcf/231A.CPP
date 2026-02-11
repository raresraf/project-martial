/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve.
 * @details This program solves the "Team" problem on Codeforces. It determines how many problems
 * a three-person team will implement based on their confidence. The team only solves a problem
 * if at least two members are sure about the solution.
 *
 * @see https://codeforces.com/problemset/problem/231/A
 *
 * Algorithm:
 * 1. Read the total number of problems, 'n'.
 * 2. Initialize a counter for solved problems, 'ans', to 0.
 * 3. Loop 'n' times, once for each problem.
 *    a. Read the confidence levels of the three team members (1 for sure, 0 for not sure).
 *    b. Sum their confidence levels.
 *    c. If the sum is 2 or greater, it means at least two members are sure. The boolean
 *       expression `(sum >= 2)` evaluates to 1, which is added to 'ans'.
 * 4. Print the final count of solved problems.
 *
 * Time Complexity: O(N), where N is the number of problems, as it requires a single pass.
 * Space Complexity: O(1), as it uses a fixed amount of memory regardless of the input size.
 */
#include <iostream>

using namespace std;

int main() {
  // n: number of problems, ans: final answer (number of problems to be solved)
  int n, ans = 0;
  cin >> n;

  /**
   * @brief Main loop to iterate through each problem.
   *
   * Pre-condition: 'n' is the number of problems to process.
   * Invariant: After each iteration, 'ans' holds the count of problems solved so far.
   */
  for (int i = 0; i < n; i++) {
    // x, y, z: confidence of the three members (1 or 0).
    int x, y, z;
    cin >> x >> y >> z;

    // Block Logic: A problem is solved if at least two members are sure.
    // The boolean expression (x + y + z >= 2) evaluates to 1 if true, 0 if false.
    // This result is directly added to the answer.
    ans += (x + y + z >= 2);
  }

  // Output the total number of problems the team will implement.
  cout << ans << endl;
  return 0;
}
