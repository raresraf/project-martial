#include <bits/stdc++.h>
#include <cstring>
using namespace std;

int main() {
  int i, c = 0;
  string a, b;
  cin >> a;
  cin >> b;
  for (i = 0; i < a.size(); i++) {
    if (a[i] >= 'A' && a[i] <= 'Z')
      a[i] = tolower(a[i]);
    ;
    if (b[i] >= 'A' && b[i] <= 'Z')
      b[i] = tolower(b[i]);
  }
  for (i = 0; i < a.size(); i++) {
    if (a[i] > b[i]) {
      c = 1;
      break;
    } else if (a[i] < b[i]) {
      c = -1;
      break;
    } else
      c = 0;
  }
  cout << c << endl;

  return 0;
}
