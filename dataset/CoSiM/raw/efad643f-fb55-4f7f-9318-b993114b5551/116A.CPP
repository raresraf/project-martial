
#include <bits/stdc++.h>
using namespace std;

int main() {
  int a, b, co = 0, res = 0;
  int no;
  cin >> no;
  while (no--) {
    cin >> a >> b;
    co -= a;
    co += b;
    if (co > res)
      res = co;
  }
  cout << res << endl;

  return 0;
}
