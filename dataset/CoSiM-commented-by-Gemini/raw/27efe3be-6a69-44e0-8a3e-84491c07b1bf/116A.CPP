/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem (116A) from Codeforces.
 * @details The program calculates the minimum required capacity of a tram by finding the
 * maximum number of passengers on board at any single point in time, given the number
 * of passengers entering and exiting at each stop.
 * @submission_details
 *   - SubmissionId: 15918867
 *   - ContestId: 116
 *   - Index: A
 *   - ProblemName: Tram
 *   - ProblemTags: ['implementation']
 *   - ProgrammingLanguage: GNU C++
 *   - Verdict: OK
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This solution simulates the passenger flow stop by stop.
 * 1. Read the total number of stops (n).
 * 2. Initialize `p` (current passengers) and `c` (required capacity) to 0.
 * 3. Loop 'n' times, once for each stop.
 * 4. In each iteration, read the number of exiting (`a`) and entering (`b`) passengers.
 * 5. Update the current passenger count: `p = p - a + b`.
 * 6. Compare the current count `p` with the max capacity `c` seen so far. If `p` is
 *    larger, update `c` to this new maximum.
 * 7. After the loop, `c` holds the minimum required capacity, which is then printed.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of stops, due to the single loop.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */

#include <bits/stdc++.h>
using namespace std;
int main() {
  // `j` is declared but never used.
  long long int n, i, j, c = 0, p = 0, a, b;
  cin >> n;
  /**
   * @block
   * @brief Simulates passenger flow over 'n' stops to find the peak occupancy.
   * @invariant `p` holds the number of people on the tram after the current stop's exchange.
   *            `c` holds the maximum value `p` has reached so far.
   * @pre 'n' is the number of stops to simulate.
   * @post 'c' contains the minimum required tram capacity.
   */
  for (i = 0; i < n; i++) {
    cin >> a >> b;
    p -= a;
    p += b;
    if (p > c)
      c = p;
  }
  cout << c;
  return 0;
}