/** 158A - Next Round
 * http://codeforces.com/problemset/problem/158/A
 * Category: Implementation
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, ans = 0, score, ref;
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> score;
    if (score > 0 && i < k - 1)
      ans++;
    else if (score > 0 && i == k - 1) {
      ans++;
      ref = score;
    } else if (score == ref)
      ans++;
    else
      break;
  }
  cout << ans << "\n";
  return 0;
}
