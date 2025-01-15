#include <bits/stdc++.h>
using namespace std;
int main() {
  int t, x = 0;
  char ch[4];
  cin >> t;
  while (t--) {
    cin >> ch;
    if ((ch[0] == '+' && ch[1] == '+') || (ch[1] == '+' && ch[2] == '+')) {
      x++;
    } else {
      x--;
    }
  }
  cout << x << endl;
}
