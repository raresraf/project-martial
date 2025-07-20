#include <bits/stdc++.h>
using namespace std;
int main() {
  int la, n, flag = 0;
  string a, b;
  cin >> a >> b;
  la = a.length();
  for (int i = 0; i < la; i++) {
    if (a[i] < 'a') {
      a[i] = a[i] + 32;
    }
    n = a[i];
    if (b[i] < 'a') {
      b[i] = b[i] + 32;
    }
    n = b[i];
    if (a[i] > b[i]) {
      cout << 1 << endl;
      break;
    }
    if (a[i] < b[i]) {
      cout << -1 << endl;
      break;
    }
    if (a[i] == b[i]) {
      flag++;
      if (flag == la) {
        cout << 0 << endl;
        break;
      }
    }
  }
}
