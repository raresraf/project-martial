/**
 * @file 1140D.CPP
 * @brief Calculates the sum of products of consecutive integers.
 *
 * This program calculates the sum of `i * (i + 1)` for `i` ranging from 2 to `n-1`,
 * where `n` is a given integer. This specific summation is often related to combinatorial
 * problems, such as counting certain configurations in polygons. For instance, it can
 * represent the sum of costs of triangulating a convex n-gon, where the cost of a
 * triangle with vertices i, j, k is i*j*k.
 *
 * @b Algorithm: The program uses a simple for-loop that iterates from `i = 2` up to `n-1`.
 * In each iteration, it calculates the product `i * (i + 1)` and adds it to a running sum.
 *
 * @b Time_Complexity: O(N), where N is the input integer. The loop runs N-2 times.
 * @b Space_Complexity: O(1), as it only uses a few variables to store the input and the sum.
 */
#include <bits/stdc++.h>
//#include <boost/multiprecision/cpp_int.hpp>

#define ll unsigned long long int
#define ld long double
#define pb push_back
#define pi 3.1415926536
#define mod 1e9 + 7
#define mp make_pair
#define wt while (t--)
#define fillarray for (int i = 0; i < n; i++)
#define testcases                                                              
  ll t;
  cin >> t;
  while (t--)

using namespace std;
// using namespace boost::multiprecision;

int main() {
  int n;
  // Read the input integer n.
  cin >> n;

  ll sum = 0;

  /**
   * @brief This loop calculates the sum of i * (i + 1) for i from 2 to n-1.
   *
   * Pre-condition: `n` is an integer greater than or equal to 3.
   * Invariant: `sum` holds the accumulated value for the iterations from 2 to `i-1`.
   */
  for (int i = 2; i < n; i++) {
    sum += i * (i + 1);
  }

  // Output the final calculated sum.
  cout << sum;
}