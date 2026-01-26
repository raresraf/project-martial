#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, a;
  int i, l;
  cin >> s;
  l = s.size();
  for (i = 0; i < l; i++) {
    if (s[i] != '+') {
      a += s[i];
    }
  }
  sort(a.begin(), a.end());
  l = a.size();
  cout << a[0];
  for (i = 1; i < l; i++) {
    cout << "+" << a[i];
  }
}
