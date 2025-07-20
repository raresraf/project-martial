#include <bits/stdc++.h>
#include <string.h>
using namespace std;
int main() {
  string a;
  int n, l;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a;
    l = a.length();
    if (l > 10) {
      cout << a[0] << l - 2 << a[l - 1] << endl;
    } else {
      cout << a << endl;
    }
  }
}
