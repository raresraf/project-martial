/**
 * @file 158A.CPP
 * @brief Logic for qualification counting in a competitive programming context.
 * 
 * Logic: Calculates the number of participants who qualify for the next round. 
 * Advancement is contingent on meeting or exceeding the score of the k-th 
 * rank finisher, provided the score is strictly positive.
 * 
 * Algorithm: Single-pass scan with threshold filtering.
 * Time Complexity: $O(N)$ where N is the participant count.
 * Space Complexity: $O(N)$ for the score buffer.
 */

#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

// Functional Utility: Redefines standard integer to 64-bit for precision safety.
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

/**
 * Functional Utility: Decouples C++ and C streams for high-throughput I/O optimization.
 */
#define fast_io()                                                              \
  cin.sync_with_stdio(false);                                                  \
  cout.sync_with_stdio(false);
#define PI 3.1415926535
#define long_zero 0ll
#define long_one 1ll

inline int sbt(int x) { return __builtin_popcount(x); }

using namespace std;
using namespace __gnu_pbds;
#define FOR(i, j, k, in) for (i = j; i < k; i += in)
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

typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    ost;

inline int mul(int a, int b, int m = mod) { return (a * b) % m; }
inline int add(int a, int b, int m = mod) { return (a + b) % m; }
inline int sub(int a, int b, int m = mod) { return (a - b + m) % m; }

int32_t main() {
  fast_io();
  int n, k;
  // Functional Utility: Reads total headcount and target qualification rank.
  cin >> n >> k;
  
  // Data Structure: Sequence of participant scores (1-based indexing used in logic).
  int a[n + 1];
  int c = 0;
  for (int i = 1; i <= n; i++) {
    cin >> a[i];
  }

  /**
   * Functional Utility: Extracts the reference score benchmark from rank k.
   */
  int reference_score = a[k];

  /**
   * Block Logic: Evaluates participants against the dual advancement constraints.
   * Condition 1: Score must be greater than or equal to the reference benchmark.
   * Condition 2: Score must be strictly positive.
   */
  for (int i = 1; i <= n; i++) {
    if (a[i] >= reference_score && a[i] > 0)
      c++;
  }
  
  // Output: Total count of qualified individuals.
  cout << c << endl;
  
  return 0;
}
