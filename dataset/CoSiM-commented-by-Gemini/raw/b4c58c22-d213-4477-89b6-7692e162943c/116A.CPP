/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem on Codeforces.
 * @url https://codeforces.com/problemset/problem/116/A
 *
 * @details
 * The problem is to calculate the minimum possible capacity of a tram.
 * We are given the number of stops and for each stop, the number of
 * passengers exiting and entering the tram. The task is to find the
 * maximum number of passengers inside the tram at any given time, which
 * corresponds to the required capacity.
 *
 * @algorithm
 * This is a simulation problem. We maintain a running count of the current
 * number of passengers in the tram and a variable to track the maximum
.
 * number of passengers seen so far. At each stop, we update the current
 * passenger count and then update the maximum if the current count is greater.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of tram stops, as we
 * iterate through each stop once.
 * Space Complexity: O(1), as we only use a few variables to keep track
 * of the passenger counts.
 */
#include <bits/stdc++.h>

#define _for(i, a, b) for (int i = (a), _b_ = (b); i < _b_; i++)
#define _fod(i, a, b) for (int i = (a), _b_ = (b); i > _b_; i--)
#define _it(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i)
#define _all(v) v.begin(), v.end()
#define __(v) memset(v, 0, sizeof(v))

using namespace std;
typedef long long LL;
typedef unsigned long long ULL;
template <typename T> vector<T> &operator+=(vector<T> &v, T x) {
  v.push_back(x);
  return v;
}

void solve() {
  int n, max_value = 0, current_value = 0, a, b;
  cin >> n;
  
  /**
   * @brief Simulate the tram's journey through all stops.
   * `a` is the number of passengers exiting, `b` is the number entering.
   * `current_value` tracks the number of passengers in the tram.
   * `max_value` tracks the maximum `current_value` seen so far.
   */
  _for(i, 0, n) {
    cin >> a >> b;
    if (!i) {
      // At the first stop, passengers only enter.
      current_value = b;
      max_value = b;
    } else {
      // At subsequent stops, passengers exit and then enter.
      current_value -= a;
      current_value += b;
      // Update the maximum capacity if needed.
      if (current_value > max_value)
        max_value = current_value;
    }
  }
  cout << max_value << endl;
}

int main() {
  // Fast I/O
  ios_base::sync_with_stdio(0);
  cin.tie(0);
#ifdef HIEUNV
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  solve();
}
