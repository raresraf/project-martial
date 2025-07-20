#include <bits/stdc++.h>

using namespace std;

int main() {
  long long n;
  int ct = 0;
  cin >> n;
  while (n > 0) {
    if (n % 10 == 4 || n % 10 == 7)
      ct++;
    n /= 10;
  }
  if (ct == 4 || ct == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
