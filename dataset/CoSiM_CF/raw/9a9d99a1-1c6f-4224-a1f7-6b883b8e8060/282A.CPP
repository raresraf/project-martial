#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, x = 0;
  string s;
  cin >> n;

  while (n--) {
    cin >> s;
    x += s[1] == '+' ? 1 : -1;
  }
  cout << x;
}