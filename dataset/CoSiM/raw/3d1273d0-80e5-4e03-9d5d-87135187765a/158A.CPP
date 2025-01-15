#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;
  int a, b, s = 0, h = 0;
  cin >> n >> k;

  for (int i = 0; i < n; i++) {

    cin >> a;
    if (a != 0) {
      h++;
    }
    if (i == k - 1) {
      b = a;
      if (b == 0) {
        cout << h;
        return 0;
      } else {
        s = k;
      }
    }
    if ((i >= k) && (a == b)) {
      s++;
    } else if (i >= k) {
      break;
    }
  }
  cout << s;

  return 0;
}