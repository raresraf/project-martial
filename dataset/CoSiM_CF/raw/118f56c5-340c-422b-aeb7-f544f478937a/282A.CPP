/** *******************
 * Ahmed El-Taweel
 * ahmed.m.eltaweel@gmail.com
 *********************************/

#include <iostream>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("c.in", "rt", stdin);
#endif

  int counter = 0;

  int n;
  string stat;

  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> stat;
    if (stat == "++X" || stat == "X++")
      counter++;
    else
      counter--;
  }

  cout << counter;

  return 0;
}
