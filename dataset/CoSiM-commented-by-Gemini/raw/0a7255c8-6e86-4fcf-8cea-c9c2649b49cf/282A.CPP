/**
 * @file 282A.CPP
 * @brief Solution for a problem involving simple increment and decrement operations.
 * @details This program calculates the final value of a variable `X` initialized to 0.
 * It reads an integer `n`, which is the number of operations. Then, it reads `n` strings,
 * each representing an operation. The operations can be "++X", "X++", "--X", or "X--".
 * The program increments a counter for "++X" and "X++", and decrements it for "--X" and "X--".
 * Finally, it prints the final value of the counter. This is a common introductory problem in competitive programming.
 *
 * @algorithm Simulation of variable modification based on string input.
 * @complexity
 * Time complexity: O(N) where N is the number of operations, as each operation is processed in constant time.
 * Space complexity: O(1) besides the storage for the input strings.
 */
#include <bits/stdc++.h>
#include <string.h>
using namespace std;
/**
 * @def IOS
 * @brief Macro for optimizing C++ standard streams I/O.
 * @details This macro unties cin from cout and sets cin, cout, and cerr to not sync with stdio,
 * which can significantly speed up I/O operations in competitive programming.
 */
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"
#define Max(x, y, z) max(x, max(y, z))
#define Min(x, y, z) min(x, min(y, z))
#define int long long
#define trace1(x) cerr << #x << ": " << x << endl
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl
#define trace3(x, y, z)                                                        \
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << endl

int32_t main() {
  IOS;
  // n: the number of statements to process.
  int n;
  cin >> n;
  // ans: the final value of X, initialized to 0.
  int ans = 0;
  string temp;
  // Predefined strings for comparison.
  string str1 = "X++";
  string str2 = "++X";
  string str3 = "X--";
  string str4 = "--X";
  /**
   * @brief Main loop to process each statement.
   * @invariant Loops `n` times. In each iteration, it reads a statement and updates `ans`.
   */
  for (int i = 0; i < n; ++i) {
    cin >> temp;
    /**
     * @brief Checks the read statement and updates the counter.
     * @pre A statement string `temp` has been read.
     * @post `ans` is incremented if `temp` is an increment operation, and decremented otherwise.
     */
    if ((temp.compare(str1) == 0) || (temp.compare(str2) == 0))
      ans++;
    else
      ans--;
  }
  // Print the final result.
  cout << ans;
  return 0;
}
