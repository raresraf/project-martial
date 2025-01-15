#include <bits/stdc++.h>
using namespace std;

int main() {

  int t, c = 0;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    if (s == "X++" || s == "++X")
      c++;
    else if (s == "X--" || s == "--X")
      c--;
  }
  cout << c << endl;

  return 0;
}
