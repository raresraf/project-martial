/**
 * @58877ec9-de61-44ff-9dfb-0a795a769a0f/231A.CPP
 * @brief This program solves the Codeforces "Team" problem (231A). It counts
 * the number of problems a three-person team will implement. A problem is
 * implemented if at least two out of three friends (Petya, Vasya, Tonya) are
 * sure about the solution. This implementation uses direct logical conditions
 * to check for the "at least two sure" criteria.
 *
 * Algorithm: Iterative processing of logical conditions for each problem.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) for storing counters and input.
 */

#include <iostream> // Includes the standard input/output stream library.
using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  int n;       // Declares 'n', representing the total number of problems in the contest.
  cin >> n;    // Reads the number of problems from standard input.
  int count = 0; // Initializes 'count' to 0, which will store the number of problems the team implements.
  
  /**
   * Block Logic: Processes 'n' problems. For each problem, it reads the certainty of three friends
   * and increments a counter if at least two friends are sure.
   * Invariant: 'count' accurately reflects the cumulative number of implemented problems after each iteration.
   */
  while (n--) { // Loop 'n' times, decrementing 'n' in each iteration.
    int p, v, t; // Declares 'p', 'v', 't' as binary indicators (0 or 1) for each friend's certainty.
    cin >> p >> v >> t; // Reads the certainty status (0 or 1) for Petya, Vasya, and Tonya for the current problem.
    
    /**
     * Block Logic: Checks if at least two friends are sure about the solution.
     * Pre-condition: 'p', 'v', 't' are 0 or 1.
     * Invariant: If any two of 'p', 'v', 't' are 1, the condition is true.
     * Inline: This compound logical OR condition checks all possible pairs of friends being sure.
     *         (Petya and Vasya) OR (Petya and Tonya) OR (Vasya and Tonya).
     */
    if ((p == 1 && v == 1) || (p == 1 && t == 1) || (v == 1 && t == 1)) {
      count++; // Increments the total count of implemented problems.
    }
  }
  cout << count; // Prints the final number of problems the friends will implement.
  // The program implicitly returns 0 on successful completion of main.
}
