/**
 * @file 1140D.CPP
 * @brief Solves the "Minimum Triangulation" problem from Codeforces (1140D).
 * @details The problem asks for the minimum total cost to triangulate a regular n-sided
 * polygon, where the cost of a triangle with vertices i, j, k is i * j * k.
 *
 * @algorithm
 * The optimal strategy to minimize the total cost is to fix one vertex (e.g., vertex 1,
 * which has the smallest label) and form all possible triangles with it. This creates
 * a "fan" triangulation. The triangles formed are (1, 2, 3), (1, 3, 4), (1, 4, 5), ...,
 * up to (1, n-1, n).
 *
 * The total cost is the sum of the products of the vertex labels for each of these triangles:
 * Cost = (1 * 2 * 3) + (1 * 3 * 4) + ... + (1 * (n-1) * n)
 *
 * This can be expressed as the summation: sum_{i=3 to n} (1 * (i-1) * i).
 *
 * The code directly implements this summation. It loops from i = 3 to n, and for each i,
 * it adds `(i-1) * i` to a running total. This correctly calculates the minimum cost.
 * Note: The file contains a large amount of unused template code common in
 * competitive programming.
 *
 * @complexity
 * Time Complexity: O(N), for the single loop from 3 to N.
 * Space Complexity: O(1).
 */
///......................................///
///......... Name : Misu Dey  ...........///
///......... University: CUET ...........///
///......... Id : 1404108     ...........///
///......................................///

#include <bits/stdc++.h>
#define ll long long
#define pb push_back
#define fr(i, a, b) for (int i = a; i <= b; i++)
#define re_fr(i, a, b) for (int i = a; i >= b; i--)
#define frl(i, a, b) for (ll i = a; i <= b; i++)
#define re_frl(i, a, b) for (ll i = a; i >= b; i--)
#define mp make_pair
#define F first
#define S second
#define MX 5000010
#define eps 1e-9
#define PI acos(-1)
#define inf 1e18
#define pll pair<ll, ll>
#define mod 1000000007
#define mod2 998244353
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define PI acos(-1)
#define fast()                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
using namespace std;
/// lower bound for sorted array
/// ll lb = lower_bound( b, b+n, 100 ) - b;
struct st {
  ll a, b;
} ST[300010];
bool cmp(st &x, st &y) { return (x.a * x.b) > (y.a * y.b); }
int main() {
  fast();
  /**
   * @variable n: The number of vertices in the regular polygon.
   */
  ll n;
  cin >> n;
  
  /**
   * @variable sum: Accumulator for the total minimum triangulation cost.
   */
  ll sum = 0;
  
  /**
   * @block
   * @brief Calculates the sum of costs for the optimal fan triangulation.
   * The loop iterates from i = 3 to n, representing the triangles (1, i-1, i).
   * The cost for each such triangle is 1 * (i-1) * i.
   */
  for (ll i = 3; i <= n; i++)
    sum += (i * (i - 1));
    
  cout << sum;
  return 0;
}
