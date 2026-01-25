#include <bits/stdc++.h>
using namespace std;
main() {
  int n;
  cin >> n;
  int c = 0;
  while (n--) {
    int a[3];
    int count = 0;
    for (int i = 0; i < 3; i++) {
      cin >> a[i];
      if (a[i] == 1)
        count++;
    }
    if (count >= 2)
      c++;
  }
  cout << c;
}
