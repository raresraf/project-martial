/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (231A) using boolean logic.
 * @details This implementation determines if at least two out of three team members
 * agree on a solution. It uses a concise boolean expression to evaluate the
 * consensus, where non-zero inputs are treated as 'true'.
 *
 * @algorithm Iteration with a direct boolean check for majority.
 * @time_complexity O(n), where 'n' is the number of problems, due to a single loop.
 * @space_complexity O(1), as storage is constant.
 */
#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;
  int ans = 0; // Counter for the number of problems the team will solve.
  
  // Block Logic: Iterate through each of the 'n' problems.
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    
    // Inline: A boolean expression is used to check for a majority. In C++, a true
    // boolean value converts to 1 when added to an integer. The expression is
    // true if any two (or all three) members agree. The final 'a && b && c'
    // check is logically redundant but does not affect the correct outcome.
    ans += (a && b || a && c || b && c || a && b && c);
  }
  
  // Output the final count.
  cout << ans;
  return 0;
}
