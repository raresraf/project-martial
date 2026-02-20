/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem by counting problems with majority agreement.
 *
 * This program calculates the number of problems a team will solve. It reads 'n',
 * the number of problems, followed by 'n' lines each containing three opinions
 * (1 for 'sure', 0 for 'not sure'). A problem is solved if at least two team
 * members are sure. This implementation processes each problem's data as it's
 * read, without storing the entire dataset in memory.
 *
 * Algorithm:
 * 1. Read the number of problems, n.
 * 2. Initialize a total solved counter 't' to 0.
 * 3. Loop 'n' times:
 *    a. Initialize a temporary counter 'co' for the current problem's votes to 0.
 *    b. Read the three opinions (a, b, c).
 *    c. Sum the opinions into 'co'.
 *    d. If the sum 'co' is 2 or more, increment the total counter 't'.
 * 4. Print the final count 't'.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1), as it only stores data for one problem at a time.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: number of problems.
  // i: loop counter.
  // co: counts 'sure' votes for a single problem.
  // a, b, c: variables for the three members' opinions.
  // t: total count of problems to be solved.
  int n, i, co = 0, a, b, c, t = 0;
  
  cin >> n; // Read the number of problems.

  /**
   * Block Logic: Main processing loop.
   * Iterates through each problem to determine if it will be solved.
   */
  for (i = 1; i <= n; i++) {
    cin >> a >> b >> c;
    
    // Sum the 'sure' votes for the current problem.
    if (a == 1)
      co++;
    if (b == 1)
      co++;
    if (c == 1)
      co++;
      
    // Pre-condition: Check if at least two members are sure.
    if (co == 2 || co == 3) {
      t++; // Increment the total solved problem count.
      co = 0;
    }
    
    // @note This `co = 0` correctly resets the vote count for the next problem.
    // The `co = 0` inside the preceding if-block is therefore redundant.
    co = 0;
  }
  
  // Print the final count.
  cout << t << endl;
}
