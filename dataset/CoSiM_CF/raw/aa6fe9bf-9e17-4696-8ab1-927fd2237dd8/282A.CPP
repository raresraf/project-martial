#include <bits/stdc++.h>

using namespace std;

int main() {
  int t;
  cin >> t;
  int a = 0;
  string s;
  while (t--) {
    cin >> s;
    if (s == "++X")
      ++a;
    else if (s == "X++")
      a++;
    else if (s == "--X")
      --a;
    else if (s == "X--")
      a--;
  }
  cout << a << "\n";
  return 0;
}
