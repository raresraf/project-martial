/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 *
 * @details
 * This program determines how many problems a three-person team will solve.
 * The team decides to solve a problem only if at least two of the three members
 * are confident in the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems (n).
 * 2. Initialize a counter for solvable problems to zero.
 * 3. Loop 'n' times, once for each problem.
 *    a. Read the three members' opinions (represented as 0 for no, 1 for yes).
 *    b. Sum the three opinions.
 *    c. If the sum is 2 or greater, it means at least two members are confident,
 *       so increment the counter.
 * 4. Print the final counter value.
 *
 * Time Complexity: O(N), where N is the number of problems, due to the single loop.
 * Space Complexity: O(1), as the memory usage is constant regardless of the input size.
 */
#include <bits/stdc++.h>

// Macro for fast I/O operations in C++. It disables synchronization with C
// stdio and unties cin from cout.
#define _                                                                      
  ios_base::sync_with_stdio(0);                                                
  cin.tie(0);

using namespace std;

int main() {
  // Apply fast I/O settings.
  _ 
  
  int n;
  cin >> n;
  
  // `ans` will store the count of problems the team will implement.
  int ans = 0;
  
  /**
   * Block Logic: Iterates through each of the 'n' problems.
   */
  while (n--) {
    // `a`, `b`, `c` represent the confidence of the three team members.
    // `x` will be the sum of their confidence values.
    int x = 0, a, b, c;
    cin >> a >> b >> c;
    
    // The sum represents how many members are sure.
    x = a + b + c;
    
    // Pre-condition: If two or more members are sure, the team solves the problem.
    if (x > 1)
      ans++;
  }
  
  // Output the total number of problems to be solved.
  cout << ans << "
";

  return 0;
}
