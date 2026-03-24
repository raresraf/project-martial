/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem involving counting problems solved by a team.
 *
 * Algorithm: Iterative conditional counting.
 * Time Complexity: O(N) where N is the number of problems.
 */

#include <cstdio>
using namespace std;

// Macros for common loop patterns in competitive programming.
#define rep(i, a, n) for (int i = a; i < n; i++)
#define erep(i, a, n) for (int i = a; i <= n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i++)

int main() {
  int n;
  int temp;
  int count;
  int solve = 0;

  /**
   * Reads the total number of problems to be evaluated.
   * Pre-condition: Standard input contains an integer representing N.
   */
  scanf("%d", &n);
  /**
   * Block Logic: Iterates through each problem to determine if it is solved.
   * Invariant: 'solve' accurately counts the problems meeting the criteria processed so far.
   */
  rep(i, 0, n) {
    count = 0;
    /**
     * Block Logic: Reads the participation status of three individuals for the current problem.
     * Invariant: 'count' accumulates the number of individuals who agree on the current problem.
     * Pre-condition: Each iteration of the outer loop (for 'n' problems) expects three integers as input.
     */
    rep(j, 0, 3) {
      scanf("%d", &temp);
      /**
       * Block Logic: Checks if a participant agrees (non-zero value).
       * Pre-condition: 'temp' contains the status of a participant.
       * Invariant: 'count' is incremented if the participant's status is affirmative.
       */
      if (temp)
        count++;
    }
    /**
     * Block Logic: Determines if the current problem is solved based on the number of agreeing participants.
     * Pre-condition: 'count' holds the number of participants who agreed for the current problem.
     * Invariant: 'solve' is incremented if 'count' meets or exceeds the threshold of 2.
     */
    if (count >= 2)
      solve++;
  }

  /**
   * Prints the total number of problems that meet the solved criteria.
   * Post-condition: The final count of solved problems is output to standard output.
   */
  printf("%d\n", solve);
}
