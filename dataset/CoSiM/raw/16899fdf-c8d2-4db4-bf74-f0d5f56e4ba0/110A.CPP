#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, rem, cnt4 = 0, cnt7 = 0;
  while (cin >> n) {
    while (n != 0) {
      rem = n % 10;

      n = n / 10;

      if (rem == 4) {
        cnt4++;
      } else if (rem == 7) {
        cnt7++;
      } else {
        continue;
      }
    }
    if ((cnt4 + cnt7) == 4 || (cnt4 + cnt7) == 7) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
    cnt4 = 0;
    cnt7 = 0;
  }
}
