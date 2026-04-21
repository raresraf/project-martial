
/**
 * @file 1343c.cpp
 * @brief Greedy approach to finding the maximum sum of a maximal-length alternating subsequence.
 * 
 * Algorithm: Single-pass greedy selection of local maxima within sign-homogeneous blocks.
 * Time Complexity: $O(N)$ per test case, where N is the length of the input array.
 * Space Complexity: $O(N)$ to store the entity sequence and the reduced alternating set.
 * Domain: Competitive Programming (Codeforces 1343C).
 */

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <forward_list>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

#define PF push_front
#define PB push_back
#define INS insert

#define FI first
#define SE second
#define all(obj) begin(obj), end(obj)
#define rall(obj) (obj).rbegin(), (obj).rend()

#define LB lower_bound
#define UB upper_bound

typedef long long ll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<set<int>> vsi;

typedef unsigned long long ull;
typedef pair<int, bool> pib;
typedef pair<ll, bool> plb;
typedef vector<vector<int>> vvi;
typedef vector<set<pii>> vspi;
typedef vector<vector<pii>> vvpi;
typedef map<int, int> mii;
typedef map<ll, ll> mll;
typedef map<char, int> mci;
typedef map<string, int> msi;
typedef map<string, string> mss;

/**
 * Functional Utility: Computes the Greatest Common Divisor using the Euclidean algorithm.
 */
ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }

/**
 * Functional Utility: Custom comparator for pairs based on the first element with optional reverse logic.
 */
template <class T1, class T2> struct cmpf {
  bool rev;
  inline bool operator()(const pair<T1, T2> &a, const pair<T1, T2> &b) const {
    return (a.first < b.first) ^ rev;
  }
  cmpf(bool b = false) { rev = b; }
};

/**
 * Functional Utility: Custom comparator for pairs based on the second element.
 */
template <class T1, class T2> struct cmps {
  bool rev;
  inline bool operator()(const pair<T1, T2> &a, const pair<T1, T2> &b) const {
    return (a.second < b.second) ^ rev;
  }
  cmps(bool b = false) { rev = b; }
};

// String conversion utilities for debugging and serialization.
string to_string(const char &c) { return string(1, c); }
string to_string(const string &s) { return '"' + s + '"'; }
string to_string(const char *s) { return to_string((string)s); }
string to_string(const bool &b) { return (b ? "true" : "false"); }
string to_string(const vector<bool> &v, const string &sep = " ") {
  string s = "[";
  for (int i = 0; i < (int)v.size(); i++) {
    if (i) {
      s += sep;
    }
    s += to_string(v[i]);
  }
  return s + "]";
}

template <size_t N> string to_string(const bitset<N> &v) {
  string s = "[";
  for (size_t i = 0; i < N; i++)
    s += v[i] ? '1' : '0';
  return s + "]";
}

template <class T1, class T2>
string to_string(const pair<T1, T2> &p, const string &sep = ",");

template <class T> string to_string(const T &v, const string &sep = " ") {
  bool first = true;
  string s = "{";
  for (const auto &x : v) {
    if (!first)
      s += sep;
    else
      first = false;
    s += to_string(x);
  }
  return s + "}";
}

template <class T>
string to_string(const T &v, const int &sz, const string &sep = " ") {
  string s = "[";
  for (int i = 0; i < sz; i++) {
    if (i) {
      s += sep;
    }
    s += to_string(v[i]);
  }
  return s + "]";
}

template <class T1, class T2>
string to_string(const pair<T1, T2> &p, const string &sep) {
  return "(" + to_string(p.first) + sep + to_string(p.second) + ")";
}

/**
 * Functional Utility: Modular exponentiation using binary exponentiation algorithm.
 */
ll binpow(const ll &x, const ll &p, const ll &mod) {
  assert(mod > 0);
  if (p == 0)
    return 1;
  if (p == 1)
    return x % mod;
  if (p & 1)
    return (binpow((x * x) % mod, p / 2, mod) * x) % mod;
  return binpow((x * x) % mod, p / 2, mod) % mod;
}

// In-place minimum/maximum selection utilities.
template <class T> void chmn(T &a, const T &b) {
  if (a > b)
    a = b;
}
template <class T> void chmx(T &a, const T &b) {
  if (a < b)
    a = b;
}

// Global constants for grid navigation and bounds.
int chr[4] = {-1, 0, 1, 0};
int chc[4] = {0, 1, 0, -1};

const ll MOD = 1e9 + 7;
const int INIT = 1001001001;
const int MN = 200001;

/**
 * Functional Utility: Optimizes standard I/O for high-performance requirements.
 */
void setIO() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
}

/**
 * @brief Checks if two integers have different mathematical signs.
 * 
 * Pre-condition: Input values 'a' and 'b' are non-zero (problem constraint).
 */
bool diff(int a, int b) { return (a > 0 && b < 0) || (a < 0 && b > 0); }

ll A[MN];

/**
 * @brief Solves a single instance of the alternating sum problem.
 * 
 * Logic:
 * 1. Groups consecutive elements with the same sign.
 * 2. Selects the largest element from each group to maximize the sum.
 * 3. Maintains the maximum possible subsequence length by picking exactly one 
 *    element per sign-homogeneous block.
 */
void solve() {
  int N;
  cin >> N;

  for (int i = 0; i < N; i++) {
    cin >> A[i];
  }

  vector<ll> num;
  /**
   * Block Logic: Maximal alternating sequence construction.
   * Pre-condition: Array 'A' is populated with 'N' elements.
   * Invariant: 'num' contains a sequence of elements with strictly alternating signs.
   */
  for (int i = 0; i < N; i++) {
    if (num.empty() || diff(num.back(), A[i])) {
      // Logic: Start a new sign-homogeneous block.
      num.push_back(A[i]);
    } else {
      // Logic: Update the current block's representative to the maximum value found.
      chmx(num.back(), A[i]);
    }
  }

  ll ans = 0;
  // Functional Intent: Aggregates the greedy optimal choices.
  for (int i = 0; i < (int)num.size(); i++) {
    ans += num[i];
  }
  cout << ans << endl;
}

/**
 * @brief Orchestrates test case execution.
 */
int main() {
  setIO();

  ll T = 1;
  cin >> T;

  while (T--)
    solve();

  return 0;
}
