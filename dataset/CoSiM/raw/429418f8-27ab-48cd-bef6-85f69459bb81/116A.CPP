/********************************
 * Ahmed El-Taweel
 * ahmed.m.eltaweel@gmail.com
 *********************************/

#include <iostream>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("c.in", "rt", stdin);
#endif

  int diff = 0;
  int max = 0;
  int n, e, o;

  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> o >> e;

    diff = diff + e - o;
    if (diff > max)
      max = diff;
  }

  cout << max << endl;

  return 0;
}
