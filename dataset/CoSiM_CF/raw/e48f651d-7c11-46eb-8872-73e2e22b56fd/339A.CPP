#include <bits/stdc++.h>
using namespace std;
int main() {
  string a;
  char temp;
  cin >> a;
  int l;
  l = a.length();
  for (int i = 0; i < l; i += 2) {
    if (a[i] != '+') {
      for (int j = i + 2; j < l; j += 2) {
        if (a[i] > a[j]) {
          temp = a[i];
          a[i] = a[j];
          a[j] = temp;
        }
      }
    }
  }
  for (int i = 0; i < l; i++) {
    cout << a[i];
  }
}
