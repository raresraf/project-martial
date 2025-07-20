#include <bits/stdc++.h>

using namespace std;

int main() {

  string s;

  cin >> s;

  int k = s.length();

  int z = 0;

  for (int i = 0; i < k; i++) {

    if (s[i] == '4' || s[i] == '7') {

      z++;
      ;
    }
  }

  if (z == 4 || z == 7) {

    cout << "YES";

  }

  else {

    cout << "NO";
  }
}