#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int i, j, k;
  while (cin >> s) {
    k = 0;
    for (i = 49; i <= 51; i++) {
      for (j = 0; j < s.size(); j++) {
        if ((int)s[j] == i) {
          if (k == 0) {
            cout << s[j];
            k = 1;
          } else
            cout << "+" << s[j];
        }
      }
    }
    cout << endl;
  }

  return 0;
}
