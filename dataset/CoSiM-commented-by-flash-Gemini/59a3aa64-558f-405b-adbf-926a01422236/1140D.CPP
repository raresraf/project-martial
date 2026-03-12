///......................................///
///......... Name : Misu Dey  ...........///
///......... University: CUET ...........///
///......... Id : 1404108     ...........///
///......................................///

/**
 * @59a3aa64-558f-405b-adbf-926a01422236/1140D.CPP
 * @brief This program calculates a specific sum for a given integer 'n'.
 * The sum is defined as the accumulation of `i * (i - 1)` for all `i` from 3 up to `n`.
 * This problem type often relates to combinatorial calculations or geometric sequences.
 *
 * Algorithm: Iterative summation.
 * Time Complexity: O(N) where N is the input integer.
 * Space Complexity: O(1) for storing variables.
 */

#include <bits/stdc++.h> // Includes most standard C++ library headers.
#define ll long long     // Defines 'll' as a shorthand for long long.
#define pb push_back     // Defines 'pb' as a shorthand for vector::push_back.
#define fr(i, a, b) for (int i = a; i <= b; i++)      // Macro for a common for-loop pattern.
#define re_fr(i, a, b) for (int i = a; i >= b; i--)   // Macro for a reverse for-loop pattern.
#define frl(i, a, b) for (ll i = a; i <= b; i++)      // Macro for long long for-loop.
#define re_frl(i, a, b) for (ll i = a; i >= b; i--)   // Macro for reverse long long for-loop.
#define mp make_pair       // Defines 'mp' as a shorthand for std::make_pair.
#define F first            // Defines 'F' as a shorthand for pair::first.
#define S second           // Defines 'S' as a shorthand for pair::second.
#define MX 5000010         // Defines MX for a common maximum value.
#define eps 1e-9           // Defines eps for a small epsilon value (floating point comparisons).
#define PI acos(-1)        // Defines PI using acos(-1.0).
#define inf 1e18           // Defines inf as a large number for infinity.
#define pll pair<ll, ll>   // Defines pll as a shorthand for pair<ll, ll>.
#define mod 1000000007     // Defines 'mod' for a common modulo value.
#define mod2 998244353     // Defines 'mod2' for another common modulo value.
#define gcd(a, b) __gcd(a, b) // Defines 'gcd' using __gcd (GCC extension).
#define lcm(a, b) (a * (b / gcd(a, b))) // Defines 'lcm' for least common multiple.
// #define PI acos(-1) // PI is defined twice.
#define fast()                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
using namespace std;
/// lower bound for sorted array
/// ll lb = lower_bound( b, b+n, 100 ) - b;
struct st { // Unused struct definition.
  ll a, b;
} ST[300010];
bool cmp(st &x, st &y) { return (x.a * x.b) > (y.a * y.b); } // Unused comparison function.
int main() {
  fast(); // Calls the fast I/O macro.
  ll n;   // Declares 'n', the input integer.
  cin >> n; // Reads 'n' from standard input.
  ll sum = 0; // Initializes 'sum' to 0, which will store the calculated result.
  /**
   * Block Logic: Calculates the sum of `i * (i - 1)` for `i` from 3 to 'n'.
   * Invariant: 'sum' accumulates the required terms in each iteration.
   */
  for (ll i = 3; i <= n; i++) // Loops from i=3 up to and including 'n'.
    sum += (i * (i - 1));     // Inline: Adds the product of 'i' and 'i-1' to 'sum'.
  cout << sum;                // Prints the final calculated sum.
  return 0;                   // Indicates successful program termination.
}
