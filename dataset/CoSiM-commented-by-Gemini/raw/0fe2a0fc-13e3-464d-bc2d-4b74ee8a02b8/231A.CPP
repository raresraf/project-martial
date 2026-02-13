/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (Codeforces 231A).
 * @author competitive-programmer
 *
 * @details This program calculates the number of problems a three-person team will
 * decide to implement. The team solves a problem only if at least two members
 * are sure about the solution (represented by 1).
 *
 * Algorithm:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for the final answer, `ans`, to zero.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, initialize a local counter `sure` to zero.
 * 5. Read the three opinions (0 or 1). Since the values are 0 or 1, they can be
 *    directly added to the `sure` counter.
 * 6. If the final sum `sure` is greater than 1 (i.e., 2 or 3), it means at
 *    least two members are sure, so increment `ans`.
 * 7. After the loop, print the final value of `ans`.
 *
 * Time Complexity: O(n), where `n` is the number of problems, as we process
 * each problem once.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // n: the number of problems. ans: the count of problems to be implemented.
  int n, ans = 0, sure;
  cin >> n;

  /**
   * Block Logic: This loop iterates 'n' times, once for each problem.
   * The `n--` condition decrements `n` on each iteration until it is zero.
   */
  while (n--) {
    int option;
    // `sure` will count the number of '1's for the current problem.
    sure = 0;

    /**
     * Logic: Reads the three opinions (0 or 1) for the current problem.
     * Since the inputs are 0 or 1, they can be summed up directly to count
     * the number of members who are sure.
     */
    cin >> option;
    sure += option;
    cin >> option;
    sure += option;
    cin >> option;
    sure += option;
    
    /**
     * Decision Logic: If the number of sure members is greater than 1 (i.e., 2 or 3),
     * the team will implement the problem.
     */
    if (sure > 1)
      ans++;
  }
  
  // Print the final count of problems to be implemented.
  cout << ans;
  return 0;
}
