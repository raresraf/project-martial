#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, tmp, mxoff = 0, mnoff = 0;
  cin >> n;

  if (n == 1) {
    cout << 0 << " " << 1 << endl;
    return 0;
  }

  mnoff = (n / 7) * 2;
  tmp = n % 7;
  if (tmp == 6)
    mnoff++;

  mxoff = 2;
  n -= 2;
  mxoff += ((n / 7) * 2);
  tmp = n % 7;
  if (tmp == 6)
    mxoff++;

  cout << mnoff << " " << mxoff << endl;

  return 0;
}
