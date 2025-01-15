#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int len, coun = 0, now, last;
  string inp;
  while (cin >> inp) {
    coun = 1;
    len = inp.size();
    for (int i = 0; i < len; i++) {

      now = inp[i];
      if (i == 0) {
        last = now;
        coun = 1;
        continue;
      }

      if (now == last) {
        coun++;
      } else {
        coun = 1;
      }
      last = now;
      if (coun == 7) {
        break;
      }
    }
    if (coun >= 7) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
}
