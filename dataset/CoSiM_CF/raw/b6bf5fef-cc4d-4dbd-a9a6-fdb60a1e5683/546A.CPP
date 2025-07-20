// http://ideone.com/BQzJDe
// 546A bananas

#include <iostream>

typedef long long ll;

using namespace std;

int main() {

  ll n, w, k, res;

  cin >> k >> n >> w;

  res = ((w * (w + 1)) / 2) * k;

  if (res <= n)
    cout << "0";
  else
    cout << res - n;

  return 0;
}
