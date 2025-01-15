#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;

  cin >> s;

  int l = s.size(), p = 0;

  char a[(l / 2) + 1];

  for (int i = 0; i < l; i++) {
    if (i % 2 == 0) {
      a[p] = s[i];
      p++;
    }
  }

  sort(a, a + (l / 2) + 1);

  for (int i = 0; i < l / 2; i++) {
    cout << a[i] << "+";
  }
  cout << a[l / 2] << endl;
}
