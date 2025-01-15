#include <bits/stdc++.h>

using namespace std;

int main() {
  int p, a[1000], b[1000], c[1000], count = 0;

  cin >> p;

  for (int i = 0; i < p; i++) {
    cin >> a[i] >> b[i] >> c[i];
  }

  for (int i = 0; i < p; i++) {
    if (a[i] == 1 && b[i] == 1)
      count++;
    else if (b[i] == 1 && c[i] == 1)
      count++;
    else if (c[i] == 1 && a[i] == 1)
      count++;
  }

  cout << count;

  return 0;
}
