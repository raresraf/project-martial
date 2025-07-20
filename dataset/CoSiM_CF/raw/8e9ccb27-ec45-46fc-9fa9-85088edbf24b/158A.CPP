#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k, a[100];
  while (cin >> n >> k) {
    int cnt = 0;
    for (int i = 0; i < n; i++) {
      cin >> a[i];
    }

    for (int i = 0; i < n; i++) {
      if (a[i] >= a[k - 1] && a[i] != 0)
        cnt++;
    }

    cout << cnt << endl;
  }

  return 0;
}
