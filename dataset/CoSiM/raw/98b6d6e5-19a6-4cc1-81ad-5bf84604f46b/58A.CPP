#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, a = 0, b = 0;
  string s, p = "hello";
  cin >> s;

  for (i = 0; i < s.length(); i++) {

    if (s[i] == p[a]) {

      a++;
      b++;
    }
  }

  if (b == 5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
