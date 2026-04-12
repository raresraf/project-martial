/**
 * @file 1343c.cpp
 * @brief Solution for a competitive programming problem.
 *
 * @author Stanley
 * @date Created on: Apr 20, 2020
 *
 * @section Problem
 * The problem is to find the maximum possible sum of a subsequence of a given
 * array, with the constraint that the signs of the elements in the subsequence
 * must alternate (positive, negative, positive, ... or negative, positive, negative, ...).
 *
 * @section Algorithm
 * The problem can be solved using a greedy approach. We can iterate through the
 * array and group consecutive numbers that have the same sign. From each of these
 * groups, we should pick the largest number to contribute to our subsequence.
 * This is because we can only pick one number from each alternating-sign block,
 * so we greedily pick the best one.
 *
 * For example, if we have a block of positive numbers `[1, 5, 2]`, followed by
 * negative numbers, we should pick `5` from the positive block.
 *
 * The overall algorithm is:
 * 1. Partition the array into contiguous sub-arrays where all elements in a
 *    sub-array have the same sign.
 * 2. Find the maximum element in each sub-array.
 * 3. The final answer is the sum of these maximum elements.
 *
 * @section Complexity
 * - Time Complexity: O(N), where N is the number of elements in the array, as
 *   we only need a single pass to identify the groups and find the maximums.
 * - Space Complexity: O(N) in the worst case for storing the chosen maximums
 *   (e.g., if the signs alternate for every element).
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

// Common macros for competitive programming for brevity.
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

ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }

// Template for custom comparators.
template <class T1, class T2> struct cmpf {
  bool rev;
  inline bool operator()(const pair<T1, T2> &a, const pair<T1, T2> &b) const {
    return (a.first < b.first) ^ rev;
  }
  cmpf(bool b = false) { rev = b; }
};
template <class T1, class T2> struct cmps {
  bool rev;
  inline bool operator()(const pair<T1, T2> &a, const pair<T1, T2> &b) const {
    return (a.second < b.second) ^ rev;
  }
  cmps(bool b = false) { rev = b; }
};

// Utility functions for debugging and string conversion.
string to_string(const char &c) { return string(1, c); }
string to_string(const string &s) { return '"' + s + '"'; }
string to_string(const char *s) { return to_string((string)s); }
string to_string(const bool &b) { return (b ? "true" : "false"); }
string to_string(const vector<bool> &v, const string &sep = " ") {
  string s = "[";
  for (int i = 0; i < v.size(); i++) {
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

// Debugging macros.
#define debug(obj) cout << #obj << ": " << to_string(obj) << endl;
#define debug1(obj, sz) cout << #obj << ": " << to_string(obj, sz) << endl;
#define debug2(obj, sz1, sz2)                                                  \
  cout << "-----" << #obj << "-----\n";                                        \
  for (int i = 0; i < sz1; i++)                                                \
    cout << to_string(obj[i], sz2) << " ";                                     \
  cout << endl;
#define debug3(obj, sz1, sz2, sz3)                                             \
  cout << "-----" << #obj << "-----\n";                                        \
  for (int i = 0; i < sz1; i++) {                                              \
    for (int j = 0; j < sz2; j++)                                              \
      cout << to_string(obj[i][j], sz3) << " ";                                \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;

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

// Shorthand for min/max updates.
template <class T> void chmn(T &a, const T &b) {
  if (a > b)
    a = b;
}
template <class T> void chmx(T &a, const T &b) {
  if (a < b)
    a = b;
}

int chr[4] = {-1, 0, 1, 0};
int chc[4] = {0, 1, 0, -1};

const ll MOD = 1e9 + 7; // 998244353;
const int INIT = 1001001001;
const int MN = 200001;

// Function to optimize standard I/O for speed.
void setIO() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
}

// Checks if two integers have different signs.
bool diff(int a, int b) { return (a > 0 && b < 0) || (a < 0 && b > 0); }

ll A[MN];

/**
 * @brief Solves a single test case for the problem.
 */
void solve() {
  int N;
  cin >> N;

  for (int i = 0; i < N; i++) {
    cin >> A[i];
  }

  // `num` will store the maximum element from each block of numbers with the same sign.
  vector<ll> num;
  for (int i = 0; i < N; i++) {
    // If `num` is empty or the current number has a different sign than the previous one,
    // start a new block.
    if (num.empty() || diff(num.back(), A[i])) {
      num.push_back(A[i]);
    } else {
      // If the sign is the same, update the last element of the block to be the maximum
      // value seen so far in that block.
      chmx(num.back(), A[i]);
    }
  }

  // The final answer is the sum of the maximums from each block.
  ll ans = 0;
  for (int i = 0; i < num.size(); i++) {
    ans += num[i];
  }
  cout << ans << endl;
}

int main() {
  setIO();

  ll T = 1;
  cin >> T;

  while (T--)
    solve();

  return 0;
}
