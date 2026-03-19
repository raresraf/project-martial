// Problem: Find the maximum sum of a subsequence with alternating signs.
// This is a competitive programming problem. The goal is to select elements from an array
// such that their signs alternate, and their sum is maximized.
// The core idea is to greedily pick the largest number from each consecutive block of numbers with the same sign.
// For example, in [1, 2, -3, -4, 5], the blocks are [1, 2], [-3, -4], and [5].
// We would pick 2, -3, and 5, and the result is 2 + (-3) + 5 = 4.
// Time Complexity: O(N) for each test case, as we iterate through the array once.
// Space Complexity: O(N) to store the input array for each test case.

#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pii;

// The following are template overloads for the << operator,
// used for debugging purposes to easily print standard containers.
template <typename T1, typename T2>
ostream &operator<<(ostream &os, pair<T1, T2> p) {
  os << p.first << " " << p.second;
  return os;
}
template <typename T> ostream &operator<<(ostream &os, vector<T> &v) {
  for (T i : v)
    os << i << ", ";
  return os;
}
template <typename T> ostream &operator<<(ostream &os, set<T> s) {
  for (T i : s)
    os << i << ", ";
  return os;
}
template <typename T1, typename T2>
ostream &operator<<(ostream &os, map<T1, T2> m) {
  for (pair<T1, T2> i : m)
    os << i << endl;
  return os;
}

int main() {
  int t;
  cin >> t;
  while (t--) {
    int n;
    cin >> n;
    int a[n];
    for (int i = 0; i < n; i++) {
      cin >> a[i];
    }
    
    // 'm' holds the maximum value in the current block of numbers with the same sign.
    int m = a[0];
    // 'sign' stores the sign of the current block (true for negative, false for positive).
    bool sign = (bool)signbit(a[0]);
    // 'ans' accumulates the sum of the chosen maximums.
    ll ans = 0;
    
    // Iterate through the array starting from the second element.
    for (int i = 1; i < n; i++) {
      // Check if the sign of the current element is different from the sign of the current block.
      if ((bool)signbit(a[i]) != sign) {
        // If the sign changes, it means a block has ended. Add the maximum element of that block to the answer.
        ans += m;
        // Reset 'm' to a very small number to find the max of the new block.
        m = INT_MIN;
        // Update the sign for the new block.
        sign = (bool)signbit(a[i]);
      }
      // Update 'm' to be the maximum element seen so far in the current block.
      m = max(m, a[i]);
    }
    // Add the maximum element from the last block to the answer.
    ans += m;
    cout << ans << endl;
  }
  return 0;
}
