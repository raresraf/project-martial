/**
 * @file 546A.CPP
 * @brief This program calculates the total cost of buying 'w' bananas, where
 *        the i-th banana costs 'i * k'. It then determines the amount of money
 *        that needs to be borrowed if the total cost exceeds the initial money 'n'.
 * Algorithm: Arithmetic calculation (sum of an arithmetic progression).
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) for storing a few integer variables.
 */
// RUMAN CSE (6th Intake) BRUR
#include <algorithm>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <strstream>
#include <vector>
#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b
#define PI 3.1416

using namespace std;
int main() {
  // freopen("input.txt","r",stdin); // Commented out: redirects standard input from "input.txt"
  // freopen("output.txt","w",stdout); // Commented out: redirects standard output to "output.txt"
  // k: cost of the first banana.
  // n: initial amount of money the person has.
  // w: number of bananas to buy.
  // cost: total calculated cost of 'w' bananas.
  // borrow: amount of money needed to borrow.
  long long k, n, w, cost, borrow;
  // Reads the cost of the first banana (k), initial money (n), and number of bananas (w).
  cin >> k >> n >> w;
  // Calculates the total cost of 'w' bananas. The sum of costs (k + 2k + ... + wk)
  // is k * (1 + 2 + ... + w) = k * (w * (w + 1) / 2).
  cost = k * ((w * (w + 1)) / 2);
  /**
   * Block Logic: Determines the amount of money to borrow.
   * Pre-condition: 'cost' and 'n' have been calculated/provided.
   * Invariant: If 'cost' is greater than 'n', 'borrow' is the difference; otherwise, 'borrow' is 0.
   */
  if (cost > n)
    borrow = cost - n; // If total cost exceeds initial money, calculate the deficit.
  else
    borrow = 0; // Otherwise, no money needs to be borrowed.
  // Prints the calculated amount to borrow.
  cout << borrow << endl;

  return 0;
}
