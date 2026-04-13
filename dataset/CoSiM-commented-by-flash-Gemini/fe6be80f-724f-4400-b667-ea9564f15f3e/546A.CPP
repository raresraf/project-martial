/**
 * @file 546A.CPP
 * @brief This program calculates the total cost of buying 'w' bananas, where
 *        the i-th banana costs 'i * k'. It then determines the amount of money
 *        that needs to be borrowed if the total cost exceeds the initial money 'n'.
 * Algorithm: Iterative summation of costs, conditional borrowing calculation.
 * Time Complexity: O(W) where W is the number of bananas, due to the loop summing costs.
 * Space Complexity: O(1) for storing a few integer variables.
 */
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define REP(i, n) for (int i = 0; i < (int)(n); ++i)
#define FOR(i, m, n) for (int i = (m); i < (int)(n); ++i)
#define ALL(x) (x).begin(), (x).end()
#define INF 2000000000

#ifdef LOCAL
#define dprintf(...) fprintf(stdout, __VA_ARGS__)
#else
#define dprintf(...) 0
#endif

using namespace std;

typedef long long LL;
typedef unsigned long long ULL;
typedef unsigned int uint;

const double PI = 3.1415926535897932384626433832795;

int main() {
  // k: cost of the first banana.
  // n: initial amount of money the person has.
  // w: number of bananas to buy.
  LL k, n, w;
  // Reads the cost of the first banana (k), initial money (n), and number of bananas (w).
  cin >> k >> n >> w;
  LL price = k; // Initial price for the first banana.
  LL totalCost = 0; // Accumulator for the total cost of bananas.
  /**
   * Block Logic: Iteratively calculates the total cost of 'w' bananas.
   * Invariant: `totalCost` accumulates the sum of `price` for each banana,
   *            and `price` is updated for the next banana (k * (i+2)).
   */
  REP(i, w) { // Loop 'w' times for each banana.
    totalCost += price; // Adds the current banana's price to the total cost.
    price = k * (i + 2); // Calculates the price for the next banana (i+2 because i starts from 0, and cost is (i+1)*k).
  }
  // Calculates the amount to borrow. If totalCost is less than or equal to n, borrow is 0.
  LL answer = totalCost - n <= 0 ? 0 : totalCost - n;
  // Prints the calculated amount to borrow.
  cout << answer << endl;
  return 0;
}
