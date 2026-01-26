#include <bits/stdc++.h>

using namespace std;

int main() {

  int x;

  cin >> x;

  int k = 0;

  for (int i = 0; i < x; i++) {

    string s;

    cin >> s;

    for (int i = 0; i < s.length(); i++) {

      if (s[i] == '+') {

        k++;

        break;
      }

      if (s[i] == '-') {

        k--;

        break;
      }
    }
  }

  cout << k;
}
