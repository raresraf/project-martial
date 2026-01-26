#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, p, v, t, cnt;
  while (cin >> n) {
    cnt = 0;
    while (n--) {
      cin >> p >> v >> t;
      if ((p == 1 && v == 1 && t == 1) || (p == 1 && v == 1) ||
          (v == 1 && t == 1) || (p == 1 && t == 1))
        cnt++;
    }
    cout << cnt << endl;
  }
  return 0;
}
