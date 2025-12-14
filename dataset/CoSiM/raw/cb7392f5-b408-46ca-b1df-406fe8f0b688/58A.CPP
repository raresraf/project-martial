#include <bits/stdc++.h>
#include <stdlib.h>
#include <string>

using namespace std;

int main() {
  string s;

  int h = 0, e = 0, l = 0, l1 = 0, o = 0;

  cin >> s;

  for (int i = 0; i < s.size(); i++) {
    if (s[i] == 'h' && h == 0) {
      h++;
    } else if (s[i] == 'e' && h == 1 && e == 0) {
      e++;
    } else if (s[i] == 'l' && e == 1 && l == 0) {
      l++;
    } else if (s[i] == 'l' && l == 1 && l1 == 0) {
      l1++;
    } else if (s[i] == 'o' && l1 == 1 && o == 0) {
      o++;
    }
  }

  if (h + e + l + l1 + o == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
