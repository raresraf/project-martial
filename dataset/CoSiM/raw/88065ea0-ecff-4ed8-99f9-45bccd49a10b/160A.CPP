#include <bits/stdc++.h>

using namespace std;

int main() {

  int x;

  cin >> x;

  int a[x];

  int k = 0, s = 0, z = 0;

  for (int i = 0; i < x; i++) {

    cin >> a[i];

    s = s + a[i];
  }

  sort(a, a + x);

  for (int i = x - 1; i >= 0; i--) {

    k = k + a[i];

    z++;

    if (k > s - k) {

      break;
    }
  }

  cout << z;
}
