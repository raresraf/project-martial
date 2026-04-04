/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem on Codeforces.
 *
 * @details A soldier wants to buy 'w' bananas. The cost of the bananas increases
 * arithmetically: the first costs 'k', the second '2k', ..., the i-th costs 'i*k'.
 * Given the initial money 'n' the soldier has, the program calculates the amount
 * he needs to borrow.
 *
 * Algorithm:
 * 1. Read the cost of the first banana 'k', the initial money 'n', and the
 *    number of bananas to buy 'w'.
 * 2. Calculate the total cost of 'w' bananas. The total cost is the sum of an
 *    arithmetic progression: k * (1 + 2 + ... + w).
 *    The code implements this summation with a loop.
 * 3. Subtract the soldier's initial money 'n' from the total cost.
 * 4. If the result is positive, that's the amount to borrow. If it's zero or
 *    negative, he doesn't need to borrow, so the amount is 0.
 * 5. Print the amount to borrow.
 *
 * Time Complexity: O(W), where W is the number of bananas.
 * Space Complexity: O(1).
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

// Macros for common loop patterns and utilities.
#define REP(i, n) for (int i = 0; i < (int)(n); ++i)
#define FOR(i, m, n) for (int i = (m); i < (int)(n); ++i)
#define ALL(x) (x).begin(), (x).end()
#define INF 2000000000

// Macro for local debugging.
#ifdef LOCAL
#define dprintf(...) fprintf(stdout, __VA_ARGS__)
#else
#define dprintf(...) 0
#endif

using namespace std;

// Typedefs for common data types.
typedef long long LL;
typedef unsigned long long ULL;
typedef unsigned int uint;

const double PI = 3.1415926535897932384626433832795;

int main() {
  LL k, n, w;
  cin >> k >> n >> w;
  LL price = k;
  LL totalCost = 0;
  /**
   * @brief Loop to calculate the total cost of 'w' bananas.
   * Logic: The loop calculates the sum of k*1 + k*2 + ... + k*w.
   * The 'price' variable is initialized to k (cost of the 1st banana) and
   * inside the loop, it's updated to hold the cost for the *next* banana,
   * which will be used in the next iteration.
   */
  REP(i, w) {
    totalCost += price;
    price = k * (i + 2);
  }
  // Calculate the amount to borrow. If totalCost is less than or equal to n, no money is needed.
  LL answer = totalCost - n <= 0 ? 0 : totalCost - n;
  cout << answer << endl;
  return 0;
}

