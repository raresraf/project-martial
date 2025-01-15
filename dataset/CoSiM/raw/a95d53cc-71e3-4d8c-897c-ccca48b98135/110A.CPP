#include <cmath>
#include <cstdio>
#include <iostream>
using namespace std;
int main() {
  long long n, a, b, cnt1, cnt2;
  cin >> n;
  cnt1 = 0, cnt2 = 0;
  while (n > 0) {
    a = n / 10;
    b = n % 10;
    n = a;

    if (b == 4 || b == 7)
      cnt1++;
    else
      cnt2++;
  }
  if (cnt1 == 4 || cnt1 == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
