#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;

  int x = 0;

  string s;
  for (int i = 0; i < n; i++) {
    cin >> s;
    if (s[0] == '+' || s[2] == '+') {
      x++;
    } else if (s[0] == '-' || s[2] == '-') {
      x--;
    }
  }

  cout << x << endl;
}