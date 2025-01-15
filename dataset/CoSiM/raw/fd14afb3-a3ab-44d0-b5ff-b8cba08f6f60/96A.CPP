#include <bits/stdc++.h>
using namespace std;
int main() {
  int l, z = 0, c = 0, o = 0;
  string a;
  cin >> a;
  l = a.length();
  for (int i = 0; i < l; i++) {
    if (a[i] == 0) {
      if (c - i == 0) {
        c++;
        z++;
        if (z >= 7) {
          cout << "Y" << endl;
          break;
        }
      }
    } else if (a[i] == 1) {
      if (c - i == 0) {
        c++;
        o++;
        if (o >= 7) {
          cout << "Y" << endl;
          break;
        }
      }
    }
  }
}
