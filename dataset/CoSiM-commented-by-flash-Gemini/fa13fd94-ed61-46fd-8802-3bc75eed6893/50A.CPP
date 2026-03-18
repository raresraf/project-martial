/**
 * @file 50A.CPP
 * @brief This program counts the number of problem solutions based on a simple majority rule.
 * For each of 'd' problems, three participants (a, b, c) vote. A problem is considered solved
 * if at least two participants vote 'yes' (represented by 1).
 * Algorithm: Iterative processing of 'd' sets of votes, summing votes, and incrementing a counter
 * if the sum meets the criteria.
 * Time Complexity: O(d) where 'd' is the number of problems.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */
#include <iostream>
using namespace std;
int main() {
  long long int a, b, c, d, coun = 0;
  ; // Deliberate empty statement; no functional utility.
  cin >> d;
  /**
   * @brief Iterates 'd' times to process the votes for each problem.
   * Pre-condition: 'd' is the total number of problems. 'coun' is initialized to 0.
   * Invariant: In each iteration, 'a', 'b', 'c' represent the votes for the current problem.
   */
  for (int i = 1; i <= d; i++) {
    cin >> a;
    cin >> b;
    cin >> c;
    /**
     * @brief Checks if at least two participants voted 'yes' for the current problem.
     * Logic: If the sum of votes (a + b + c) is 2 or more, the problem is considered solved.
     * Post-condition: If the condition is met, 'coun' is incremented.
     */
    if (a + b + c >= 2)
      coun++;
  }
  // Outputs the final count of solved problems. Note: The post-increment operator
  // `coun++` will increment 'coun' after its value is used in the output operation.
  cout << coun++;
}
