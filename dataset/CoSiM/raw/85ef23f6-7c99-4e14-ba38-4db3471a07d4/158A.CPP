#include <bits/stdc++.h>
using namespace std;
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  int k, n, i;
  cin >> n >> k;
  int a[n], cmp, count = 0, flag = 0;
  for (i = 0; i < n; ++i) {
    cin >> a[i];
    if (a[i] != 0) {
      if (count < k)
        ++count;
      if (count == k and flag == 0) {
        cmp = a[i];
        flag = 1;
        continue;
      }
      if (flag == 1 and a[i] == cmp)
        ++count;
      // cout<<flag<<" "<<cmp<<" "<<count<<"\n";
    }
  }
  if (a[0] == 0)
    cout << 0 << "\n";
  else
    cout << count << "\n";
  return 0;
}
