/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem.
 * @details This program calculates the final value of a variable 'X' after a series of increment
 * or decrement operations. The operations are given as strings ("++X", "X++", "--X", "X--").
 * The program cleverly inspects the second character of the string to determine the operation.
 *
 * Algorithm: Simple Counter Simulation
 * The program initializes a counter to zero. It then iterates through each operation string,
 * incrementing the counter if the string contains a '+' and decrementing if it contains a '-'.
 *
 * Time Complexity: O(N), where N is the number of operations, as it processes each string once.
 * Space Complexity: O(1), as it uses a fixed amount of memory regardless of the input size.
 */

#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

// Defines 'int' as 'long long' for handling larger integer values, a common practice in competitive programming.
#define int long long
#define ld long double
#define mod 1000000007
#define mod1 mod
#define mod2 100000009
#define show(a)                                                                \
  for (int i = 0; i < a.size(); i++)                                           \
    cout << a[i & 1] << " ";
#define fi first
#define se second
#define vi vector<int>
#define vs vector<string>
#define vll vector<ll>
#define pb push_back
#define pi pair<int, int>
#define si set<int>
#define sll set<ll>
#define pll pair<ll, ll>
#define sci(x) scanf("%d", &x)
#define scs(x) scanf("%s", x)
#define scll(x) scanf("%I64d", &x)
#define all(v) v.begin(), v.end()
#define sot(v) sort(v.begin(), v.end())
#define maxheap priority_queue<int>
#define mp make_pair
// Optimizes C++ stream I/O for speed.
#define fast_io()                                                              \
  cin.sync_with_stdio(false);                                                  \
  cout.sync_with_stdio(false);
#define PI 3.1415926535
#define long_zero 0ll
#define long_one 1ll

// Unused helper function to count set bits.
inline int sbt(int x) { return __builtin_popcount(x); }

using namespace std;
using namespace __gnu_pbds;

// Unused generic for-loop macro.
#define FOR(i, j, k, in) for (i = j; i < k; i += in)

// Unused debugging macro infrastructure.
#define TRACE
#ifdef TRACE
#define deb(...) __f(#__VA_ARGS__, __VA_ARGS__)
template <typename Arg1> void __f(const char *name, Arg1 &&arg1) {
  cout << name << " : " << arg1 << std::endl;
}
template <typename Arg1, typename... Args>
void __f(const char *names, Arg1 &&arg1, Args &&...args) {
  const char *comma = strchr(names + 1, ',');
  cout.write(names, comma - names) << " : " << arg1 << " | ";
  __f(comma + 1, args...);
}
#else
#define deb(...)
#endif
#define N 10003
bool ok;

// Unused policy-based data structure (ordered set).
typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    ost;

// Unused modular arithmetic helper functions.
inline int mul(int a, int b, int m = mod) { return (a * b) % m; }
inline int add(int a, int b, int m = mod) { return (a + b) % m; }
inline int sub(int a, int b, int m = mod) { return (a - b + m) % m; }

int32_t main() {
  // Functional Utility: Decouples C++ streams from C streams for faster I/O.
  fast_io();
  
  // n: the number of statements to execute.
  // i: loop counter.
  // sum: the variable representing 'X', initialized to 0.
  int n, i, sum = 0;
  // s: stores the operation string.
  string s;
  
  // Read the total number of operations.
  cin >> n;

  /**
   * Block Logic: Process each of the 'n' operations.
   * Pre-condition: 'n' holds the number of operations to read.
   * Invariant: 'sum' holds the accumulated value after the first 'i' operations.
   */
  for (i = 0; i < n; i++) {
    cin >> s;
    // Core Logic: Inspect the second character of the operation string.
    // For "++X", "X++", it's '+'. For "--X", "X--", it's '-'.
    // This ternary operator increments 'sum' by 1 for '+' and decrements by 1 for '-'.
    sum += s[1] == '+' ? 1 : -1;
  }
  
  // Output the final value of the counter.
  cout << sum;
}
