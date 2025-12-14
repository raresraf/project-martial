#include <iostream>
using namespace std;
int main() {
  int k, n, w, a;
  cin >> k >> n >> w;
  a = ((((w * (w + 1)) / 2) * k) - n);
  if (a > 0)
    cout << a << endl;
  else
    cout << 0 << endl;
  return 0;
}
