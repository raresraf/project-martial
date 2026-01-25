#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, x = 0;

  cin >> n;

  string s;

  while (n--) {
    cin >> s;
    if (s == "++X" || s == "X++")
      x++;
    else
      x--;
  }
  cout << x << endl;

  return 0;
}
