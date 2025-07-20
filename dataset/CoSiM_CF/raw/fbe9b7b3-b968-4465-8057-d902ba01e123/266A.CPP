#include <iostream>
using namespace std;
#define min(a, b) ((a) < (b) ? true : false)
int main() {
  ios_base::sync_with_stdio(false);
  long n, m;
  double a, b;
  long res = 0;
  cin >> n >> m >> a >> b;
  if (min((b / m), a)) {
    res = (n / m) * b;
    n %= m;
    if (n * a < b)
      res += n * a;
    else
      res += b;
  } else {
    res = a * n;
  }
  cout << res;
  return 0;
}