// Team
//
// https://codeforces.com/problemset/problem/231/A

// One day three best friends Petya, Vasya and Tonya decided to form a team and
// take part in programming contests. Participants are usually offered several
// problems during programming contests. Long before the start the friends
// decided that they will implement a problem if at least two of them are sure
// about the solution. Otherwise, the friends won't write the problem's solution.
//
// This contest offers n problems to the participants. For each problem we know,
// which friend is sure about the solution. Help the friends find the number of
// problems for which they will write a solution.
//
// Input
// The first input line contains a single integer n (1   n   1000)   the number
// of problems in the contest. Then n lines contain three integers each, each
// integer is either 0 or 1. If the first number in the line equals 1, then Petya
// is sure about the problem's solution, otherwise he isn't sure. The second
// number shows Vasya's view on the solution, the third number shows Tonya's
// view. The numbers on the lines are separated by spaces.
//
// Output
// Print a single integer   the number of problems the friends will implement on
// the contest.

/**
 * @57f0d0be-7f91-44c2-87cb-72796feaefa4/231A.CPP
 * @brief This program solves the Codeforces "Team" problem (231A). It counts
 * the number of problems a three-person team (Petya, Vasya, Tonya) will
 * implement, based on the rule that at least two members must be sure about
 * the solution for a problem to be implemented.
 *
 * Algorithm: Iterative counting with majority vote for each problem.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) for storing counters and input.
 */

#include <iostream> // Includes the standard input/output stream library.

using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  int n;       // Declares 'n', representing the total number of problems in the contest.
  int p, v, t; // Declares 'p', 'v', 't' as binary indicators (0 or 1) for each friend's certainty.
  p = v = t = 0; // Initializes friends' certainty to 0 (not sure).
  int ans = 0;   // Initializes 'ans' to 0, which will count the problems the team implements.
  cin >> n;      // Reads the number of problems from standard input.

  /**
   * Block Logic: Iterates through each of the 'n' problems to determine if the team will implement it.
   * Invariant: 'ans' accurately reflects the count of implemented problems from previous iterations.
   */
  for (int i = 0; i < n; i++) {
    int count = 0; // Initializes 'count' to 0 for the current problem, to tally friends who are sure.
    cin >> p >> v >> t; // Reads the certainty status (0 or 1) for Petya, Vasya, and Tonya for the current problem.
    
    // Block Logic: Tallies the number of friends who are sure about the solution.
    // Invariant: 'count' stores the number of friends (0 to 3) who are sure for the current problem.
    if (p == 1) // Inline: If Petya is sure (p=1), increment count.
      count++;
    if (v == 1) // Inline: If Vasya is sure (v=1), increment count.
      count++;
    if (t == 1) // Inline: If Tonya is sure (t=1), increment count.
      count++;

    /**
     * Block Logic: Checks if the problem meets the implementation criteria.
     * Pre-condition: 'count' holds the total number of friends who are sure about the current problem.
     * Invariant: 'ans' is incremented if 'count' is greater than 1 (at least two friends are sure).
     */
    if (count > 1) // If at least two friends are sure.
      ans++;       // Increment the total count of implemented problems.
  }
  cout << ans << endl; // Prints the final number of problems the friends will implement.
  return 0;            // Indicates successful program termination.
}
