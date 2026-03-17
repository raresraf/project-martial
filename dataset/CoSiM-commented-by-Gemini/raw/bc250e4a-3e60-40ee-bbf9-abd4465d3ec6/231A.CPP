/**
 * @file 231A.CPP
 * @brief Solution to the Codeforces problem 231A "Team".
 * @source http://codeforces.com/problemset/problem/231/A
 * 
 * @details
 * This program calculates the number of problems a three-person team will solve in a
 * programming contest. The team decides to implement a problem only if at least two
 * members are confident about the solution.
 *
 * Algorithm:
 * The solution iterates through each problem one by one. For each problem, it sums up the
 * confidence levels of the three members (represented by 1 for confident, 0 for not).
 * If the sum is 2 or greater, it means there is a majority consensus, and a counter for
 * solved problems is incremented.
 *
 * Time Complexity: O(N), where N is the total number of problems. The program processes
 * each problem in a single pass.
 * Space Complexity: O(1), as the memory usage is constant regardless of the number of problems.
 */

// http://codeforces.com/problemset/problem/231/A
// SOLUTION BY DN54321

#include <iostream>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  int n;
  // Read the total number of problems.
  scanf("%d", &n);
  
  int res = 0, i, j, k; // `res` will store the count of problems to be solved. 
  
  /**
   * Block Logic: Iterate through each of the 'n' problems.
   * Invariant: At the beginning of each iteration, 'res' holds the count of
   * problems solved from the previous iterations.
   */
  for (int l = 0; l < n; ++l) {
    // Read the confidence levels of the three team members for the current problem.
    scanf("%d%d%d", &i, &j, &k);
    
    /**
     * Block Logic: Check for majority consensus.
     * The sum of confidence levels (i + j + k) will be 2 or 3 if at least two members
     * are confident (value of 1). This is a concise way to check the condition.
     */
    if (i + j + k > 1)
      ++res;
  }
  
  // Output the total number of problems the team will solve.
  printf("%d\n", res);
  return 0;
}
