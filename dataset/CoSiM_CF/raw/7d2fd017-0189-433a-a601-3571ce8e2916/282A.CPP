#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, x = 0;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    if (s == "X++")
      x++;
    else if (s == "++X")
      ++x;
    else if (s == "X--")
      x--;
    else if (s == "--X")
      --x;
  }
  cout << x << "\n";
}
