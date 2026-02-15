/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem using a standard competitive programming template.
 * @details The problem is to determine if a body is in equilibrium, which occurs if the
 * vector sum of all forces acting on it is the zero vector (0, 0, 0).
 *
 * @section algorithm Algorithm
 * This program calculates the net force by summing the components of all force vectors.
 * 1. Global variables `a`, `b`, and `c` are used to maintain the running sum of forces
 *    along the X, Y, and Z axes, respectively.
 * 2. The main function reads the number of force vectors, `tc`.
 * 3. It then loops `tc` times, calling a `solve()` function in each iteration.
 * 4. The `solve()` function is responsible for reading the three components of a single
 *    force vector and adding them to the global sums.
 * 5. After the loop, the main function checks if the global sums `a`, `b`, and `c` are
 *    all zero.
 * 6. It prints "YES" if the net force is zero, indicating equilibrium, and "NO" otherwise.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of force vectors, as each vector is processed once.
 * - Space Complexity: O(1), as a fixed number of variables are used for the sums.
 */

#include <bits/stdc++.h>
using namespace std;

// This section contains a typical competitive programming template with type aliases
// and macros to speed up coding. For example, `ll` is `long long`, `vi` is `vector<int>`,
// and `FOR` is a loop macro.
typedef long long ll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef pair<string, string> pss;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<pii> vii;
typedef vector<ll> vl;
typedef vector<vl> vvl;

double EPS = 1e-9;
int INF = 1000000005;
long long INFF = 1000000000000000005ll;
double PI = acos(-1);
int dirx[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
int diry[8] = {0, 1, -1, 0, -1, 1, -1, 1};
const ll MOD = 1000000007;

#define DEBUG fprintf(stderr, "====TESTING====\n")
#define VALUE(x) cerr << "The value of " << #x << " is " << x << endl
#define OUT(x) cout << x << endl
#define debug(...) fprintf(stderr, __VA_ARGS__)
#define READ(x)                                                                \
  for (auto &(z) : x)                                                          \
    cin >> z;
#define FOR(a, b, c) for (int(a) = (b); (a) < (c); ++(a))
#define FORN(a, b, c) for (int(a) = (b); (a) <= (c); ++(a))
#define FORD(a, b, c) for (int(a) = (b); (a) >= (c); --(a))
#define FORSQ(a, b, c) for (int(a) = (b); (a) * (a) <= (c); ++(a))
#define FORC(a, b, c) for (char(a) = (b); (a) <= (c); ++(a))
#define EACH(a, b) for (auto &(a) : (b))
#define REP(i, n) FOR(i, 0, n)
#define REPN(i, n) FORN(i, 1, n)
#define Mp(a, b) a = mp(a, b)
#define MIN(a, b) a = min(a, b)
#define SQR(x) ((ll)(x) * (x))
#define RESET(a, b) memset(a, b, sizeof(a))
#define fi first
#define se second
#define mp make_pair
#define pb push_back
#define ALL(v) v.begin(), v.end()
#define ALLA(arr, sz) arr, arr + sz
#define SIZE(v) (int)v.size()
#define SORT(v) sort(ALL(v))
#define REVERSE(v) reverse(ALL(v))
#define SORTA(arr, sz) sort(ALLA(arr, sz))
#define REVERSEA(arr, sz) reverse(ALLA(arr, sz))
#define PERMUTE next_permutation
#define TC(t) while (t--)
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL)

// Global variables to store the sum of forces along each axis (X, Y, Z).
int a = 0, b = 0, c = 0;
// Global variables used to read the components of a single vector.
int x, y, z;

/**
 * @brief Reads one force vector from input and adds its components to the global sums.
 */
void solve() {
  cin >> x >> y >> z;
  a += x;
  b += y;
  c += z;
}

int main() {
  FAST_INP;
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int tc; // 'tc' (test cases) is used here to mean the number of vectors.
  cin >> tc;
  // The TC macro expands to `while(tc--)`, processing one vector per loop.
  TC(tc) solve();
  
  // After all vectors are processed, check if the total force is zero.
  // The OUT macro expands to `cout << ... << endl`.
  OUT((a == 0 && b == 0 && c == 0 ? "YES" : "NO"));
  return 0;
}