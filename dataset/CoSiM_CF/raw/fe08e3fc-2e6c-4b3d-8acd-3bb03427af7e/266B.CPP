#include <bits/stdc++.h>

using namespace std;

int main() {

  int x;

  cin >> x;

  int y;

  cin >> y;

  string s;

  cin >> s;

  int k = s.length();

  while (y > 0) {

    for (int i = 0; i < k - 1; i++) {

      if (s[i] == 'B') {

        if (s[i] != s[i + 1]) {

          s[i] = 'G';

          s[i + 1] = 'B';

          i++;
        }
      }
    }

    y--;
  }

  cout << s;
}