#include <bits/stdc++.h>
#define lli long long int
#define ye cout << "YES" << endl
#define no cout << "NO" << endl
using namespace std;

int main() {
  lli n, a, b;
  cin >> n >> a >> b;
  if (n == a) {
    ye;
    cout << 1 << ' ' << 0 << endl;
  } else if (n == b) {
    ye;
    cout << 0 << ' ' << 1 << endl;
  } else if (n % a == 0) {
    ye;
    cout << n / a << ' ' << 0 << endl;
  } else if (n % b == 0) {
    ye;
    cout << 0 << ' ' << n / b << endl;
  } else {
    lli t1 = 0;

    while (true) {
      lli num = n - b * t1;
      if (num % a == 0 && num >= 0) {
        ye;
        cout << num / a << ' ' << t1 << endl;
        break;
      } else if (num >= 0) {
        t1++;
      } else {
        no;
        break;
      }
    }
  }

  return 0;
}
