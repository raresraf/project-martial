#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int cs, p, v, t, cou = 0, cnt = 0;
  while (cin >> cs) {
    cnt = 0;
    for (int i = 0; i < cs; i++) {
      cou = 0;
      cin >> p >> v >> t;

      if (p == 1)
        cou++;
      if (v == 1)
        cou++;
      if (t == 1)
        cou++;

      if (cou >= 2)
        cnt++;
    }
    cout << cnt << endl;
  }
}
