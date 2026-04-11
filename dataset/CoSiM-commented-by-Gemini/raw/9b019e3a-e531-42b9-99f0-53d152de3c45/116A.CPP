/**
 * @file 116A.CPP
 * @author ANMOL GUPTA
 * @brief Solution for a competitive programming problem (Codeforces 116A "Tram").
 * @details This program calculates the minimum required capacity for a tram by simulating
 * passenger flow across several stops. It tracks the maximum number of passengers
 * inside the tram at any given moment.
 *
 * @note The vast majority of this file consists of a personal boilerplate
 * template for competitive programming, including complex debugging macros,
 * type definitions, and I/O optimizations. The core problem-solving logic
 * is confined to the `main` function.
 */
/*
Author:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   ANMOL GUPTA      %%%%%%%%
%%%%%%%%   NIT HAMIRPUR     %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/
#include <bits/stdc++.h>
using namespace std;
// --- BEGIN BOILERPLATE ---
#pragma GCC target("avx2")
#pragma GCC optimization("unroll-loops")
#pragma GCC optimize("O2")
#define pb push_back
#define debug(x) cout << '>' << #x << " : " << x << endl;
#define mp make_pair
#define ff first
#define ss second
#define allunique(v) v.erase(unique(v.begin(), v.end()), v.end());
#define MOD 1000000007
#define MOD9 1000000009
#define th(n) cout << n << endl
#define gc getchar_unlocked
#define ms(s, n) memset(s, n, sizeof(s))
#define prec(n) fixed << setprecision(n)
#define eps 0.000001
#define bolt ios::sync_with_stdio(0)
#define forr(i, n) for (ll i = 0; i < n; i++)
#define eof (scanf("%d", &n)) != EOF
#define PI acos(-1.0)
#define inf INT_MAX
#define all(v) v.begin(), v.end()
#define endl "
"
typedef long long ll;
typedef double db;
typedef long long ll;
typedef pair<ll, ll> pll;
typedef vector<ll> vll;
typedef map<ll, ll> mll;
typedef set<ll> sll;
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
// --- END BOILERPLATE ---

/**
 * @brief Main function containing the core logic for the "Tram" problem.
 * @details This function reads the number of tram stops and then iterates through
 * each stop, updating the current passenger count and tracking the maximum
 * count observed.
 *
 * @algorithm Simulation with Maximum Tracking
 * @time_complexity O(N), where N is the number of stops, because it processes each
 *                  stop in constant time.
 * @space_complexity O(1), as it only uses a few variables to store the current
 *                   and maximum passenger counts.
 */
int main() {
  // Functional Utility: Optimizes I/O operations.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  ll n, x, max_x, a, b;
  cin >> n; // Read the total number of stops.

  x = 0;     // Variable to track the current number of passengers on the tram.
  max_x = 0; // Variable to track the maximum number of passengers at any point.

  /**
   * @block
   * @brief Simulates passenger flow stop by stop.
   * @details In each iteration, it reads the number of passengers exiting (a) and
   * entering (b), updates the current passenger count, and then updates the
   * maximum capacity required so far.
   * @invariant `x` holds the current passenger count after the exchange at a stop.
   * @invariant `max_x` holds the peak passenger count observed up to the current stop.
   */
  while (n--) {
    cin >> a >> b; // Read passengers exiting (a) and entering (b).
    x += (b - a); // Update the current passenger count.
    max_x = max(x, max_x); // Update the maximum capacity required.
  }

  // `th` is a macro for `cout << n << endl`. Print the final result.
  th(max_x);
}
