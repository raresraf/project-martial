#include <bits/stdc++.h>

using namespace std;

int main() {

  int k;

  cin >> k;

  string s;

  cin >> s;

  int z = 0;

  for (int i = 0; i < k; i++) {

    if (s[i] == s[i + 1]) {

      z++;
    }
  }

  cout << z;
}