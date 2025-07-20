#include <bits/stdc++.h>
using namespace std;
bool compare(pair<int, int> p1, pair<int, int> p2) {
  if (p1.first > p2.first)
    return true;
  if (p1.first == p2.first && p1.second < p2.second)
    return true;
  return false;
}
int main() {
  int n, m;
  cin >> n >> m;
  if (n == 1) {
    cout << 1;
    return 0;
  }
  pair<int, int> a[n + 1];
  for (int i = 1; i <= n; i++) {
    cin >> a[i].first >> a[i].second;
  }
  sort(a + 1, a + n + 1, compare);
  int k = 0;
  for (int i = 2; i < n + 1; i++) {
    int f = 1;
    int flag = 0;
    while (a[i - 1].first == a[i].first && a[i - 1].second == a[i].second) {
      // cout<<a[i-1].first<<" "<<a[i].first<<" "<<a[i-1].second<<"
      // "<<a[i].second<<"\n";
      i++;
      f++;
      if (i == n + 1) {
        flag = 1;
        break;
      }
    }
    // cout<<f<<" "<<i<<"\n";
    if (i > m) {
      cout << f << "\n";
      return 0;
    }
    if (i == n && i == m && flag) {
      cout << f << "\n";
      return 0;
    }
  }
  cout << 1 << "\n";
}
