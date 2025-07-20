#include <bits/stdc++.h>

using namespace std;

int main() {
  string a, b;
  int ct = 0;
  cin >> a >> b;
  for (int i = 0; i < a.length(); i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }
  for (int i = 0; i < a.length(); i++) {
    if (a[i] == b[i])
      ct++;
    else if (a[i] > b[i]) {
      cout << 1 << endl;
      break;
    } else {
      cout << -1 << endl;
      break;
    }
  }
  if (ct == a.size())
    cout << 0 << endl;
  return 0;
}
