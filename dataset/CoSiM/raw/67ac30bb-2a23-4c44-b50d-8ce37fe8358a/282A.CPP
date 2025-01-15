#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  int n, c(0);
  cin >> n;

  string s;

  while (n--) {

    cin >> s;
    if (s == "X++" || s == "++X")
      c++;

    if (s == "--X" || s == "X--")
      c--;
  }
  cout << c << "\n";
}
