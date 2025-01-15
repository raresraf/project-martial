#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, b, i, j, l, o = 0, k;
  string a;
  cin >> a;
  for (i = 0; i < a.size(); i++) {
    l = 0;
    for (j = i + 1; j < a.size(); j++) {
      if (a[i] != a[j]) {
        l = l + 1;
      }
    }
    if (l == (a.size() - i - 1)) {
      o = o + 1;
    }
  }
  if (o % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}
