#include <bits/stdc++.h>
using namespace std;
int main() {
  long long n;
  cin >> n;
  int lucky = 0;
  while (n > 0) {
    int rem = n % 10;
    if (rem == 4 or rem == 7)
      lucky += 1;
    n = n / 10;
  }
  if (lucky == 4 or lucky == 7) {
    cout << "YES" << '\n';
  } else
    cout << "NO" << '\n';
  return 0;
}