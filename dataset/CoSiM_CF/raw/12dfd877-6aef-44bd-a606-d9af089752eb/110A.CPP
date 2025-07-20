#include <bits/stdc++.h>
using namespace std;
int main() {
  long long n;
  int f4 = 0, f7 = 0, count = 0;
  cin >> n;
  while (n) {
    if (n % 10 == 4) {
      n /= 10;
      f4 = 1;
      count++;
    } else if (n % 10 == 7) {
      n /= 10;
      f7 = 1;
      count++;
    } else {
      n /= 10;
    }
  }
  if ((f4 == 1 || f7 == 1) && (count == 4 || count == 7))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}