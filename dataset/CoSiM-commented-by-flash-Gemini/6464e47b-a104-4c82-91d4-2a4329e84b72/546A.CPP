//
//
#include <iostream>
using namespace std;
int main() {
  long long int k, n, w, i, j, sum = 0;
  cin >> k >> n >> w;
  sum = (k * w * (w + 1)) / 2;
  long long int ans = sum - n;
  // cout<<ans;
  if (ans < 0)
    cout << "0";
  else
    cout << ans;
  return 0;
}
