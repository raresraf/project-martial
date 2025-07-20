#include <bits/stdc++.h>
using namespace std;
main() {
  int n;
  cin >> n;
  char a[n];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  int count = 0;
  int j = 0;
  for (int i = 1; i < n; i++) {
    if (a[j] == a[i]) {
      count++;

    } else {
      j = i;
    }
  }
  cout << count;
}
