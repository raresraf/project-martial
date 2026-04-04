/**
 * @file 231A.CPP
 * @brief Counts problems solved by a team based on individual agreement.
 *
 * This program simulates a problem-solving scenario where for each of 't' problems,
 * three individuals (friends) cast their votes (1 for agree, 0 for disagree).
 * A problem is considered solved if at least two out of the three individuals agree.
 * The program calculates and outputs the total number of solved problems.
 *
 * Algorithm: Iterative processing of problem votes. For each problem, it sums the votes
 *            and checks if the sum is greater than or equal to 2.
 * Time Complexity: O(T), where T is the number of problems, as each problem's votes
 *                  are processed in constant time.
 * Space Complexity: O(T) for storing the counts of agreements per problem in a vector.
 */
#include <bits/stdc++.h> // Includes most standard libraries.
using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: Reads problem data, counts solved problems, and outputs the total.
   * @return 0 if the program executes successfully.
   */
  int t; // Variable to store the number of problems.
  cin >> t; // Read the number of problems.
  vector<int> cnt(t); // Vector to store the count of agreements for each problem.
  int ans = 0; // Variable to store the final count of solved problems.
  /**
   * @brief Loop to read decisions for each problem.
   * @pre 't' contains the total number of problems.
   * @post 'cnt' vector is populated with the number of agreements for each problem.
   * @invariant 't' decreases in each iteration, and problem decisions are processed one by one.
   */
  while (t--) { // Loop 't' times.
    int ct = 0; // Counter for friends who agree on the current problem.
    /**
     * @brief Inner loop to read decisions from three friends for the current problem.
     * @pre 'ct' is initialized to 0.
     * @post 'ct' holds the number of friends who agree on the current problem.
     * @invariant 'i' iterates from 0 to 2, representing each of the three friends.
     */
    for (int i = 0; i < 3; i++) { // Read 3 decisions.
      int q; // Variable to store a friend's decision (1 for agree, 0 for disagree).
      cin >> q; // Read friend's decision.
      /**
       * @brief Conditional statement to increment 'ct' if the friend agrees.
       */
      if (q == 1)
        ct++; // Increment count if friend agrees.
    }
    // Store the count of agreements for the current problem in the 'cnt' vector.
    // Note: 't' is already decremented by the while loop, so it acts as a reverse index.
    cnt[t] = ct;
  }
  /**
   * @brief Loop to count how many problems were solved (at least 2 friends agreed).
   * @pre 'cnt' vector is populated with agreement counts for all problems.
   * @post 'ans' holds the total number of solved problems.
   * @invariant 'i' iterates through the 'cnt' vector.
   */
  for (auto i : cnt) // Iterate through the vector of agreement counts.
    /**
     * @brief Conditional statement to increment 'ans' if a problem is solved.
     */
    if (i >= 2) // If at least two friends agreed.
      ans++;    // Increment the solved problems counter.
  cout << ans << endl; // Output the total number of solved problems.
  return 0;
}