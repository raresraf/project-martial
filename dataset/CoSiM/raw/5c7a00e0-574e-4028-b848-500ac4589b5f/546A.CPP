#include <iostream>
using namespace std;

int main() {
  long long int k, n, w, ans;
  cin >> k >> n >> w;
  ans = (k + w * k) * w / 2 - n;
  if (ans < 0)
    ans = 0;
  cout << ans << endl;
  return 0;
}
