
/*                     */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int t, i = 0;
  string s;

  cin >> t;

  while (t--) {
    cin >> s;
    if (s == "++X" || s == "X++")
      i++;
    else if (s == "--X" || s == "X--")
      i--;
  }
  cout << i << endl;

  return 0;
}
