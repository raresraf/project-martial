/**
 * @file 1366A.CPP
 * @brief Solution for Codeforces Problem 1366A - "Shovels and Swords".
 * @author The solution is submitted by a competitive programmer.
 *
 * @section Algorithm
 * The problem asks for the maximum number of tools (shovels or swords) that can be crafted
 * given 'a' sticks and 'b' diamonds.
 * - A shovel requires 2 sticks and 1 diamond.
 * - A sword requires 1 stick and 2 diamonds.
 *
 * Let 'x' be the number of shovels and 'y' be the number of swords. We want to maximize k = x + y.
 * The constraints are:
 * 1. 2*x + y <= a  (sticks constraint)
 * 2. x + 2*y <= b  (diamonds constraint)
 * 3. x >= 0, y >= 0
 *
 * @subsection Derivation
 * By adding the two inequalities, we get:
 * (2*x + y) + (x + 2*y) <= a + b
 * 3*x + 3*y <= a + b
 * 3*(x + y) <= a + b
 * x + y <= (a + b) / 3
 *
 * This implies that the total number of tools, k = x + y, is at most (a + b) / 3.
 *
 * Additionally, to make 'k' tools, we need at least 'k' sticks and 'k' diamonds in total,
 * regardless of the combination. For example, if we try to make 'a' tools, we would need at
 * least 'a' sticks. Thus, the total number of tools cannot exceed 'a' or 'b'.
 * k <= a
 * k <= b
 *
 * Combining these constraints, the maximum number of tools 'k' is limited by the minimum of
 * these three values.
 *
 * Time Complexity: O(1) for each test case.
 * Space Complexity: O(1) for each test case.
 */
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;

// Performance Directives: Compiler-specific optimizations for speed.
#pragma GCC target("avx2")
#pragma GCC optimization("unroll-loops")
#pragma GCC optimize("O2")

// --- Competitive Programming Boilerplate ---
// Section: Macro definitions for common functions and data types.
// These are used to shorten code and speed up typing during contests.

#define pb emplace_back
#define mp make_pair
#define ff first
#define ss second
#define endl "
"
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()
#define th(n) cout << n << endl
#define gc getchar_unlocked
#define ms(s, n) memset(s, n, sizeof(s))
#define prec(n) fixed << setprecision(n)
// Python-like aliases
#define gcd __gcd
#define append push_back
#define str to_string
#define stringtoll stoll
#define upper(s) transform(s.begin(), s.end(), s.begin(), ::toupper)
#define lower(s) transform(s.begin(), s.end(), s.begin(), ::tolower)
// Utility function macros
#define max3(a, b, c) max(a, max(b, c))
#define min3(a, b, c) min(a, min(b, c))
#define sswap(a, b)                                                            
  {                                                                            
    a = a ^ b;                                                                 
    b = a ^ b;                                                                 
    a = a ^ b;                                                                 
  }
#define swap(a, b)                                                             
  {                                                                            
    auto temp = a;                                                             
    a = b;                                                                     
    b = temp;                                                                  
  }
#define set0(dp) memset(dp, 0, sizeof(dp));
#define setbits(x) __builtin_popcountll(x)
#define zrobits(x) __builtin_ctzll(x)
#define SORT(v) sort(all(v))
#define forr(i, n) for (ll i = 0; i < n; i++)
#define formatrix(i, n) for (ll i = 0; i < n; i++, cout << "
")
#define eof (scanf("%d", &n)) != EOF
// Type aliases
#define vll vector<ll>
#define vvl vector<vector<ll>>
#define vpll vector<pair<ll, ll>>
#define pll pair<ll, ll>
#define ppl pair<ll, pll>
#define ull unsigned long long
#define ll long long
#define mll map<ll, ll>
#define sll set<ll>
// Macro for handling multiple test cases
#define w(x)                                                                   
  int x;                                                                       
  cin >> x;                                                                    
  while (x--)
#define uni(v) v.erase(unique(v.begin(), v.end()), v.end());
#define ini(a, v) memset(a, v, sizeof(a));

// Policy-Based Data Structure from GNU PBDS
template <typename T>
using MinPriorityQueue = priority_queue<T, vector<T>, greater<T>>;
typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    pbds;
// member functions :
// 1. order_of_key(k) : number of elements strictly lesser than k
// 2. (iterator)find_by_order(k) : k-th element in the set

/**
 * @brief A struct to accelerate I/O operations by decoupling C++ streams
 * from C standard streams.
 */
struct _ {
  ios_base::Init i;
  _() { ios_base::sync_with_stdio(false), cin.tie(NULL), cout.tie(NULL); }
} _;

// --- Constants Section ---
constexpr int dx[] = {-1, 0, 1, 0, 1, 1, -1, -1};
constexpr int dy[] = {0, -1, 0, 1, 1, -1, 1, -1};
constexpr ll inf = 1999999999999999997;
constexpr auto PI = 3.14159265358979323846L;
constexpr auto oo = numeric_limits<int>::max() / 2 - 2;
constexpr auto eps = 1e-6;
constexpr auto mod = 1000000007;
constexpr auto MOD = 1000000007;
constexpr auto MOD9 = 1000000009;
constexpr auto maxn = 200006;

// --- Debugging Template ---
// This section contains an advanced, variadic template for debugging,
// allowing for pretty-printing of various data structures.
#define n_l '
'
#define dbg(...)                                                               
  cout << "[" << #__VA_ARGS__ << "]: ";                                        
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
#define dbgm(...)                                                              
  cout << "[" << #__VA_ARGS__ << "]: ";                                        
  dbgs(__VA_ARGS__);                                                           
  cout << endl;

/**
 * @brief Main function to solve the problem for multiple test cases.
 */
int main() {
  // Handles 't' test cases from standard input.
  w(t) {
    ll a, b;
    cin >> a >> b; // Read the number of sticks (a) and diamonds (b).

    /**
     * @brief Calculates the maximum number of tools.
     * This is the core logic derived from the problem constraints.
     * The maximum number of tools is limited by:
     * 1. The total items available: (a + b) / 3
     * 2. The number of sticks: a
     * 3. The number of diamonds: b
     */
    cout << min({(a + b) / 3, a, b}) << endl;
  }
}
