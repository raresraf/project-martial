/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (231A) from Codeforces.
 * @details The program counts how many problems a three-person team will solve,
 * where a problem is solved if at least two members are confident.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution iterates through each problem, summing the confidence
 * levels of the team members to see if it meets the threshold.
 * 1. Read the total number of problems (n).
 * 2. Initialize a counter `c` to 0 for the problems to be solved.
 * 3. Loop 'n' times.
 * 4. In each iteration, read the three confidence levels (0 or 1) for that problem.
 * 5. Sum the levels. If the sum is greater than 1 (i.e., 2 or 3), it means at
 *    least two members are confident.
 * 6. If the condition is met, increment the counter `c`.
 * 7. After the loop, print the final count `c`.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, because of the single loop.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, c = 0;
  cin >> n;
  int p, v, t; // Variables for the three members' opinions.
  /**
   * @block
   * @brief Processes each of the 'n' problems to see if the team will solve it.
   * @invariant 'c' holds the count of problems solved so far.
   * @pre 'n' is the total number of problems.
   * @post 'c' holds the total count of problems where at least two members are confident.
   */
  for (int i = 0; i < n; i++) {
    cin >> p >> v >> t;
    // Check if the sum of confidences is 2 or more.
    if (p + v + t > 1)
      c++;
  }
  cout << c << endl;

  return 0;
}
