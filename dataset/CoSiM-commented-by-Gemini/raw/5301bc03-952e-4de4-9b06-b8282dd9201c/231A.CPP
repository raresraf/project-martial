/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The problem is to count how many problems a team of three members
 * will solve. The condition for solving a problem is that at least two of the
 * three members are sure about the solution (represented by 1).
 *
 * @algorithm
 * The number of confident members can be found by simply summing their confidence
 * values (0 or 1).
 * 1. Read the total number of problems, n.
 * 2. Initialize a result counter `r` to 0.
 * 3. Loop n times for each problem.
 *    a. Read the confidence levels of the three members (x, y, z).
 *    b. Calculate their sum `s = x + y + z`.
 *    c. If the sum `s` is 2 or greater, increment the result counter `r`.
 * 4. After the loop, print the final result `r`.
 *
 * @complexity
 * Time Complexity: O(n), for the single loop through n problems.
 * Space Complexity: O(1).
 */

// .... .... .... !

#include <fstream>
#include <iomanip>
#include <iostream>

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>

#include <cstdio>
#include <cstring>

using namespace std;

// Common competitive programming macros. 'rep' is used in this code.
#define rep(i, n) for (int i = 0, _n = (int)(n); i < _n; i++)
#define fer(i, x, n) for (int i = (int)(x), _n = (int)(n); i < _n; i++)
#define rof(i, n, x) for (int i = (int)(n), _x = (int)(x); i-- > _x;)
#define fch(i, x) for (__typeof(x.begin()) i = x.begin(); i != x.end(); i++)
#define sz(x) (int((x).size()))
#define X first
#define Y second

typedef long long ll;
typedef pair<int, int> pii;

////////////////////////////////////////////////////////////////////////////////

int main() {
  // Fast I/O for C++ streams.
  ios::sync_with_stdio(false);
  
  int n; // Number of problems.
  cin >> n;
  
  int r = 0; // The result counter for problems that will be solved.
  
  /**
   * @block
   * @brief Loop through each problem using the 'rep' macro.
   */
  rep(i, n) {
    int s, x, y, z;
    cin >> x >> y >> z;
    // Sum the confidence levels.
    s = x + y + z;
    // If two or more members are confident, increment the result.
    if (s >= 2)
      r++;
  }
  
  cout << r << endl;
  
  /**
   * @note This block appears to be for pausing the program in a local
   * testing environment, as it waits for additional input after the
   * correct answer has already been printed. It is not part of the
   * solution logic.
   */
  {
    int _;
    cin >> _;
  }
  return 0;
}
