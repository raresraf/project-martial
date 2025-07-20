#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, ans = 0;
  cin >> n;
  while (n--) {
    int a[3], count = 0;
    for (int i = 0; i < 3; i++) {
      cin >> a[i];
      if (a[i] == 1)
        count++;
    }
    if (count >= 2)
      ans++;
  }
  cout << ans;

  return 0;
}
