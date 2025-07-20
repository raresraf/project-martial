// http://codeforces.com/problemset/problem/231/A

#include <iostream>
using namespace std;

int main() {
  int n, t, sure, notSure, res = 0;
  cin >> n;
  while (n--) {
    sure = notSure = 0;
    for (int i = 1; i < 4; i++) {
      cin >> t;
      if (t)
        sure++;
      else
        notSure++;
    }
    if (sure > 1)
      res++;
  }
  cout << res;

  return 0;
}
