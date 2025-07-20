#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, i, j = 0;
  cin >> n;
  string a;
  cin >> a;
  for (i = 0; i < n - 1; i++) {
    if (a[i] == a[i + 1])
      j++;
  }
  cout << j << endl;

  return 0;
}
