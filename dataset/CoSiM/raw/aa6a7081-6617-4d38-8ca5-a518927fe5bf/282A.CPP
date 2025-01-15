#include <bits/stdc++.h>
using namespace std;
int main() {
  int x = 0, n;
  string str;
  cin >> n;
  while (n) {
    cin >> str;
    if (str == "++X")
      ++x;
    else if (str == "X++")
      x++;
    else if (str == "X--")
      x--;
    else if (str == "--X")
      --x;

    n--;
  }
  cout << x;
}
