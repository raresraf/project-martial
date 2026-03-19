/**
 * @file 546A.CPP
 * @brief Solution to Codeforces Problem 546A: "Soldier and Bananas".
 * Calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of each banana increases incrementally (1st banana costs k, 2nd costs 2k, ..., w-th costs w*k).
 * Algorithm: Calculate the total cost of 'w' bananas using the sum of an arithmetic series,
 * then subtract the soldier's initial money to find the amount to borrow.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <bits/stdc++.h>
#include <string.h> // Though not directly used in this solution, often included in competitive programming templates.
using namespace std;

// Macro for optimizing C++ standard streams for competitive programming.
// Disables synchronization with C's stdio and unties cin/cout.
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n" // Macro for newline, potentially faster than std::endl.
// Macros for finding max/min of three numbers.
#define Max(x, y, z) max(x, max(y, z))
#define Min(x, y, z) min(x, min(y, z))
#define int long long // Type alias: uses long long for int to prevent overflow in competitive programming.
// Debugging macros for printing variable names and their values to cerr.
#define trace1(x) cerr << #x << ": " << x << endl
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl
#define trace3(x, y, z)                                                        \
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << endl

/**
 * @brief Main function to solve the "Soldier and Bananas" problem.
 * Reads the cost of the first banana (k), initial money (n), and desired bananas (w).
 * Computes the total cost and the amount to borrow, then prints the result.
 */
int32_t main() {
  IOS; // Apply I/O optimizations.
  int k, n, w, ans; // 'k': cost of 1st banana, 'n': initial money, 'w': number of bananas, 'ans': total cost.
  
  // Reads the input values.
  cin >> k >> n >> w;
  
  // Calculates the total cost of 'w' bananas.
  // The sum of costs is k + 2k + ... + w*k = k * (1 + 2 + ... + w).
  // The sum 1 + 2 + ... + w is w*(w+1)/2.
  ans = k * w * (w + 1) / 2;
  
  /**
   * Block Logic: Determines the amount of money the soldier needs to borrow.
   * If the total calculated cost is less than or equal to the money the soldier has,
   * no money needs to be borrowed (print 0). Otherwise, print the difference.
   */
  if (ans < n) {
    cout << "0"; // Soldier has enough money or more.
  } else {
    cout << ans - n; // Soldier needs to borrow this amount.
  }

  return 0;
}