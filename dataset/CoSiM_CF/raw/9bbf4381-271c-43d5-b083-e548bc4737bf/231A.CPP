#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int ans = 0;
  while (n--) {
    int a, b, c;
    int cnt[3] = {0};
    cin >> a >> b >> c;
    cnt[a]++;
    cnt[b]++;
    cnt[c]++;
    if (cnt[1] >= 2)
      ans++;
  }
  cout << ans << endl;
  return 0;
}