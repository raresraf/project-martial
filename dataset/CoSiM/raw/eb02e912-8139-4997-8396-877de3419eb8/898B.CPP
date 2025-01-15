#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, a, b, f = 0;
  cin >> n >> a >> b;
  for (int y = 0; y < (n / b) + 1; y++) {
    if ((n - b * y) % a == 0 && n > b * y) {
      cout << "YES" << endl;
      cout << (n - b * y) / a << " " << y;
      f = 1;
      break;
    }
  }
  if (f == 0) {
    if (n % b == 0) {
      cout << "YES" << endl;
      cout << "0"
           << " " << n / b;
    } else
      cout << "NO" << endl;
  }
}
