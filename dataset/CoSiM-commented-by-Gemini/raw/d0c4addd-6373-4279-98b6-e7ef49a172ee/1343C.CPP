/**
 * @file 1343C.CPP
 * @brief Solves the "Alternating Subsequence" problem with a data propagation method.
 *
 * This file contains a solution to find the maximum sum of an alternating subsequence.
 * The implementation is heavily stylized with a competitive programming template, including
 * many macros and helper functions for container printing and other shortcuts.
 *
 * @b Algorithm: Multi-Pass Maximum Propagation
 *   This is an unusual, in-place approach that uses multiple passes to find the solution:
 *   1. A forward pass propagates the prefix maximum within each same-signed block.
 *   2. A backward pass propagates the suffix maximum within each same-signed block. After
 *      these two passes, every element in a block is equal to that block's maximum value.
 *   3. A final summation pass adds up the maximums. It does this by adding an element
 *      only when it differs from the previous one, which only occurs at the boundary
 *      between two blocks of different signs.
 *
 * @b Time_Complexity: O(N) per test case, due to three separate linear passes.
 * @b Space_Complexity: O(N) per test case to store the input array.
 */
#include <bits/stdc++.h>
using namespace std;
// --- Start of Competitive Programming Template ---
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
typedef long long ll;
#define int ll
typedef vector<int> vi;
typedef vector<long long> vl;
typedef pair<int, int> pi;
#define trace(x) cout << #x << "=" << x << "\n";
#define sz(x) (int)(x.size())
#define mod 1000000007
#define endl "\n"
template <typename F, typename S>
ostream &operator<<(ostream &os, const pair<F, S> &p) {
  return os << "(" << p.first << ", " << p.second << ")";
}

template <typename T> ostream &operator<<(ostream &os, const vector<T> &v) {
  os << "{";
  typename vector<T>::const_iterator it;
  for (it = v.begin(); it != v.end(); it++) {
    if (it != v.begin())
      os << ", ";
    os << *it;
  }
  return os << "}";
}

template <typename T> ostream &operator<<(ostream &os, const set<T> &v) {
  os << "[";
  typename set<T>::const_iterator it;
  for (it = v.begin(); it != v.end(); it++) {
    if (it != v.begin())
      os << ", ";
    os << *it;
  }
  return os << "]";
}

template <typename F, typename S>
ostream &operator<<(ostream &os, const map<F, S> &v) {
  os << "[";
  typename map<F, S>::const_iterator it;
  for (it = v.begin(); it != v.end(); it++) {
    if (it != v.begin())
      os << ", ";
    os << it->first << " = " << it->second;
  }
  return os << "]";
}

#define deb(x) cerr << #x << " = " << x << endl;
// --- End of Competitive Programming Template ---

int32_t main() {
  ios::sync_with_stdio(false);
  int t;
  cin >> t;
  vi a;
  // set<int>S;
  while (t--) {
    // chck for n=1
    int n;
    cin >> n;
    a.clear();
    a.resize(n);
    REP(i, 0, n) { cin >> a[i]; }

    /**
     * @brief First pass (forward): Propagates prefix max within same-sign blocks.
     */
    for (int i = 1; i < n; i++) {
      if (a[i] < 0 && a[i - 1] < 0) {
        if (a[i - 1] > a[i]) {
          a[i] = a[i - 1];
        }
      }
      if (a[i] > 0 && a[i - 1] > 0) {
        if (a[i - 1] > a[i]) {
          a[i] = a[i - 1];
        }
      }
    }

    /**
     * @brief Second pass (backward): Propagates suffix max within same-sign blocks.
     * After both passes, all elements in a block are equal to the block's maximum.
     */
    for (int i = n - 2; i >= 0; i--) {
      if (a[i] < 0 && a[i + 1] < 0) {
        if (a[i + 1] > a[i]) {
          a[i] = a[i + 1];
        }
      }
      if (a[i] > 0 && a[i + 1] > 0) {
        if (a[i + 1] > a[i]) {
          a[i] = a[i + 1];
        }
      }
    }

    ll sum = 0;
    /**
     * @brief Third pass (summation): Sums the unique values from the modified array.
     * An element is added only when it differs from the previous one, which occurs
     * at the boundary between two blocks.
     */
    for (int i = 0; i < n; i++) {
      if (i == 0) {
        sum += a[i];
      } else {
        if (a[i] != a[i - 1]) {
          sum += a[i];
        }
      }
    }
    cout << sum << "\n";
  }

  return 0;
}