#include <iostream>
using namespace std;
int main() {
  int i, c, m = 0, n = 0, a[1005], b[1005];
  cin >> c;
  for (i = 0; i < c; i++) {
    cin >> a[i] >> b[i];
    n += b[i];
    n -= a[i];
    if (n > m) {
      m = n;
    }
  }
  cout << m << endl;
}
