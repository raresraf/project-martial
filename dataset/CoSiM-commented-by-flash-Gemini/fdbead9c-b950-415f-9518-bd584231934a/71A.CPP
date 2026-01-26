#include <bits/stdc++.h>
using namespace std;

int main() {
  int t, i;
  cin >> t;
  while (t--) {
    string a;
    cin >> a;
    if (a.size() > 10) {
      cout << a[0] << a.size() - 2 << a[a.size() - 1] << endl;
    } else {
      for (i = 0; i < a.size(); i++) {
        cout << a[i];
      }
      cout << endl;
    }
  }
}