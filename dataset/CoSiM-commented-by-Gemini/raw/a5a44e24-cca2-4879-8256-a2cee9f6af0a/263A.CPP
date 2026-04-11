/**
 * @file 263A.CPP
 * @brief Solution for a competitive programming problem (Codeforces 263A "Beautiful Matrix").
 * @author Mehul Chaturvedi, IIT-Guwahati
 *
 * @details
 * This program calculates the minimum number of moves required to move a single '1'
 * in a 5x5 matrix to the center position (3,3). A move consists of swapping
 * the '1' with an adjacent element (up, down, left, or right).
 *
 * @algorithm
 * The problem is equivalent to finding the Manhattan distance between the position
 * of the '1' and the center of the matrix. The algorithm iterates through the
 * 5x5 grid to find the row (p) and column (q) of the '1'. The center of the
 * 0-indexed grid is at (2,2). The number of moves is then calculated as
 * `abs(2 - p) + abs(2 - q)`.
 *
 * @time_complexity O(1), as the matrix size is fixed at 5x5 (25 elements).
 * @space_complexity O(1), as it only uses a few variables to store the coordinates.
 *
 * @note The file includes a large template of helper functions, data structures,
 * and macros commonly used in competitive programming, though most are not used
 * for this specific problem.
 */
/*

                Name: Mehul Chaturvedi
                IIT-Guwahati

*/

/*
                Talent is Overrated
*/

#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;
template <typename T>
using ordered_set =
    tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#include <chrono>
#ifndef mehul
#pragma GCC optimize("Ofast")
#endif

typedef long long ll;
typedef long double ld;
typedef unordered_map<int, int> umapii;
typedef unordered_map<int, bool> umapib;
typedef unordered_map<string, int> umapsi;
typedef unordered_map<string, string> umapss;
typedef map<string, int> mapsi;
typedef map<pair<int, int>, int> mappiii;
typedef map<int, int> mapii;
typedef pair<int, int> pii;
typedef pair<long long, long long> pll;
typedef unordered_set<int> useti;

#define debug(x) cout << '>' << #x << ':' << x << endl;
#define uset unordered_set
#define it iterator
#define mp make_pair
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define f first
#define s second

#define INF 4557430888798830399ll
#define MOD 1000000007
#define EPS 1e-7
#define PI acos(-1)

#define sz(x) (int)(x).size()
template <typename T, typename U> inline bool exist(T &cont, U &val) {
  return cont.find(val) != cont.end();
}
#define fr(i, a, b) for (int i = (a), _b = (b); i <= _b; i++)
#define rep(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define repr(i, n) for (int i = n; i >= 0; i--)
#define frr(i, a, b) for (int i = (a), _b = (b); i >= _b; i--)
#define trav(a, x) for (auto &a : x)
#define fil(ar, val)                                                           
  memset(ar, val,                                                              
         sizeof(ar)) // 0x3f for inf, 0x80 for -INF can also use with pairs

#ifdef mehul
template <typename T> void __p(T a) { cout << a << " "; }
template <typename T> void __p(std::vector<T> a) {
  cout << "{ ";
  for (auto p : a)
    __p(p);
  cout << "}";
}
template <typename T, typename F> void __p(pair<T, F> a) {
  cout << "{ ";
  __p(a.first);
  __p(a.second);
  cout << "}";
}
template <typename T, typename F> void __p(std::vector<pair<T, F>> a) {
  cout << "{ ";
  for (auto p : a)
    __p(p);
  cout << "}";
}
template <typename T, typename... Arg> void __p(T a1, Arg... a) {
  __p(a1);
  __p(a...);
}
template <typename Arg1> void __f(const char *name, Arg1 &&arg1) {
  cout << name << " : ";
  __p(arg1);
  cout << endl;
}
template <typename Arg1, typename... Args>
void __f(const char *names, Arg1 &&arg1, Args &&...args) {
  int bracket = 0, i = 0;
  for (;; i++)
    if (names[i] == ',' && bracket == 0)
      break;
    else if (names[i] == '(')
      bracket++;
    else if (names[i] == ')')
      bracket--;
  const char *comma = names + i;
  cout.write(names, comma - names) << " : ";
  __p(arg1);
  cout << "| ";
  __f(comma + 1, args...);
}
#define trace(...)                                                             
  cout << "Line:" << __LINE__ << " ";                                          
  __f(#__VA_ARGS__, __VA_ARGS__)
int begtime = clock();
#define end_routine()                                                          
  cout << "

Time elapsed: " << fixed                                        
       << double(clock() - begtime) * 1000 / CLOCKS_PER_SEC                    
       << setprecision(12) << " ms

";
#else
#define endl '
'
#define trace(...)
#define end_routine()
#endif

// Custom hash for unordered map
struct custom_hash {
  static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
  }

  size_t operator()(uint64_t x) const {
    static const uint64_t FIXED_RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    return splitmix64(x + FIXED_RANDOM);
  }
};

// Power Function O(log(n))
ll poww(ll a, ll b, ll mod) {
  if (b == 0)
    return 1;
  ll ans = poww(a, b / 2, mod);
  if (b % 2 == 0)
    return (ans * ans) % mod;
  return (((ans * ans) % mod) * a) % mod;
}

/**
 * @brief Solves a single test case for the "Beautiful Matrix" problem.
 */
void solve() {
  ll p, q; // Variables to store the 0-indexed coordinates of the '1'.

  // Block Logic: Iterate through the 5x5 matrix to find the '1'.
  // The loops are 0-indexed (i from 0 to 4, j from 0 to 4).
  rep(i, 5) {
    rep(j, 5) {
      ll a;
      cin >> a;
      if (a == 1) {
        p = i; // Store row index.
        q = j; // Store column index.
      }
    }
  }

  // Calculate the Manhattan distance to the center (2,2).
  // The number of row moves is |2 - p| and column moves is |2 - q|.
  ll ans = abs(2 - p) + abs(2 - q);
  cout << ans << '
';
}

/**
 * @brief Main function, entry point of the program.
 */
int main(int argc, char **argv) {
  // Functional Utility: Decouples C++ streams from C streams for faster I/O.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
#ifdef mehul
  // Enables reading from a file instead of stdin for local debugging.
  freopen("input.txt", "r", stdin);
#endif

  ll t = 1;
  while (t--) {
    solve();
  }

#ifdef mehul
  end_routine();
#endif

  return 0;
}
