/**
 * @8c51aaa4-6d91-474b-9867-0fe9f06709ee/266B.CPP
 * @brief Discrete-time simulation of a queue reordering process based on local adjacency rules.
 * Domain: Competitive Programming, Algorithm Simulation.
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(T \times N)$ where T is the time duration and N is the queue length.
 * Space Complexity: $O(N)$ to store the entity sequence.
 */

#include <bits/stdc++.h>
using namespace std;

// Optimisations: Compiler directives to enhance execution speed for large-scale simulations.

#pragma GCC target("avx2")
#pragma GCC optimization("unroll-loops")
#pragma GCC optimize("O2")

// shortcuts for functions

#define pb push_back
#define mp make_pair
#define ff first
#define ss second
#define endl "\n"
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()
#define th(n) cout << n << endl
#define gc getchar_unlocked
#define ms(s, n) memset(s, n, sizeof(s))
#define prec(n) fixed << setprecision(n)
#define n_l '\n'

// make it python: Aliases to emulate higher-level language syntax.

#define gcd __gcd
#define append push_back
#define str to_string
#define upper(s) transform(s.begin(), s.end(), s.begin(), ::toupper)
#define lower(s) transform(s.begin(), s.end(), s.begin(), ::tolower)
#define print(arr)                                                             \
  for (auto el : arr)                                                          \
    cout << el << " ";                                                         \
  cout << endl

// utility functions shortcuts

#define max3(a, b, c) max(a, max(b, c))
#define min3(a, b, c) min(a, min(b, c))
#define sswap(a, b)                                                            \
  {                                                                            \
    a = a ^ b;                                                                 \
    b = a ^ b;                                                                 \
    a = a ^ b;                                                                 \
  }
#define swap(a, b)                                                             \
  {                                                                            \
    auto temp = a;                                                             \
    a = b;                                                                     \
    b = temp;                                                                  \
  }
#define init(dp) memset(dp, -1, sizeof(dp));
#define set0(dp) memset(dp, 0, sizeof(dp));
#define bits(x) __builtin_popcount(x)
#define SORT(v) sort(all(v))
#define endl "\n"
#define forr(i, n) for (ll i = 0; i < n; i++)
#define eof (scanf("%d", &n)) != EOF

// declaration shortcuts

#define vi vector<int>
#define vll vector<ll>
#define vvi vector<vector<int>>
#define vvl vector<vector<ll>>
#define pll pair<ll, ll>
#define ppl pair<ll, pp>
#define ull unsigned long long
#define ll long long
#define mll map<ll, ll>
#define sll set<ll>
#define uni(v) v.erase(unique(v.begin(), v.end()), v.end());
#define ini(a, v) memset(a, v, sizeof(a))

// Constants

constexpr int dx[] = {-1, 0, 1, 0, 1, 1, -1, -1};
constexpr int dy[] = {0, -1, 0, 1, 1, -1, 1, -1};
constexpr ll INF = 1999999999999999997;
constexpr int inf = INT_MAX;
constexpr int MAXSIZE = int(1e6) + 5;
constexpr auto PI = 3.14159265358979323846L;
constexpr auto oo = numeric_limits<int>::max() / 2 - 2;
constexpr auto eps = 1e-6;
constexpr auto mod = 1000000007;
constexpr auto MOD = 1000000007;
constexpr auto MOD9 = 1000000009;

// Debugging utilities for recursive and multi-dimensional data structure visualization.

// For reference: https://codeforces.com/blog/entry/65311

#define dbg(...)                                                               \
  cout << "[" << #__VA_ARGS__ << "]: ";                                        \
  cout << to_string(__VA_ARGS__) << endl
template <typename T, size_t N> int SIZE(const T (&t)[N]) { return N; }
template <typename T> int SIZE(const T &t) { return t.size(); }
string to_string(string s, int x1 = 0, int x2 = 1e9) {
  return '"' + ((x1 < s.size()) ? s.substr(x1, x2 - x1 + 1) : "") + '"';
}
string to_string(const char *s) { return to_string((string)s); }
string to_string(bool b) { return (b ? "true" : "false"); }
string to_string(char c) { return string({c}); }
template <size_t N> string to_string(bitset<N> &b, int x1 = 0, int x2 = 1e9) {
  string t = "";
  for (int __iii__ = min(x1, SIZE(b)), __jjj__ = min(x2, SIZE(b) - 1);
       __iii__ <= __jjj__; ++__iii__) {
    t += b[__iii__] + '0';
  }
  return '"' + t + '"';
}
template <typename A, typename... C>
string to_string(A(&v), int x1 = 0, int x2 = 1e9, C... coords);
int l_v_l_v_l = 0, t_a_b_s = 0;
template <typename A, typename B> string to_string(pair<A, B> &p) {
  l_v_l_v_l++;
  string res = "(" + to_string(p.first) + ", " + to_string(p.second) + ")";
  l_v_l_v_l--;
  return res;
}
template <typename A, typename... C>
string to_string(A(&v), int x1, int x2, C... coords) {
  int rnk = rank<A>::value;
  string tab(t_a_b_s, ' ');
  string res = "";
  bool first = true;
  if (l_v_l_v_l == 0)
    res += n_l;
  res += tab + "[";
  x1 = min(x1, SIZE(v)), x2 = min(x2, SIZE(v));
  auto l = begin(v);
  advance(l, x1);
  auto r = l;
  advance(r, (x2 - x1) + (x2 < SIZE(v)));
  for (auto e = l; e != r; e = next(e)) {
    if (!first) {
      res += ", ";
    }
    first = false;
    l_v_l_v_l++;
    if (e != l) {
      if (rnk > 1) {
        res += n_l;
        t_a_b_s = l_v_l_v_l;
      };
    } else {
      t_a_b_s = 0;
    }
    res += to_string(*e, coords...);
    l_v_l_v_l--;
  }
  res += "]";
  if (l_v_l_v_l == 0)
    res += n_l;
  return res;
}
void dbgs() { ; }
template <typename Heads, typename... Tails> void dbgs(Heads H, Tails... T) {
  cout << to_string(H) << " | ";
  dbgs(T...);
}

#define dbgm(...)                                                              \
  cout << "[" << #__VA_ARGS__ << "]: ";                                        \
  dbgs(__VA_ARGS__);                                                           \
  cout << endl;
#define n_l '\n'

int main() {
  /**
   * Functional Utility: Optimized I/O stream synchronization for competitive constraints.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  // printf("%I64d", n)
  ll n, ans = 0, t;
  cin >> n >> t;
  string s;
  cin >> s;

  /**
   * Block Logic: Orchestrates temporal progression of the queue state.
   * Invariant: At start of step 'i', queue reflects state after 'i' seconds.
   */
  for (int i = 0; i < t; i++) {
    /**
     * Block Logic: Single-pass sweep to resolve local reordering rules.
     * Logic: Swaps 'B' (Boy) and 'G' (Girl) if they are adjacent and 'B' precedes 'G'.
     */
    for (int j = 0; j < n - 1; j++) {
      if (s[j] == 'B' and s[j + 1] == 'G') {
        swap(s[j], s[j + 1]);
        // Inline: Skips next index to prevent multi-step migration in a single time unit.
        j++;
      }
    }
  }
  // Output final converged or time-bounded state.
  th(s);
}
