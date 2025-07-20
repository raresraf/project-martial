#include <bits/stdc++.h>
using namespace std;
#define MAX 100005
#define pi acos(-1.0)
typedef long long ll;

int main() {
  ll k, n, w, result;
  cin >> k >> n >> w;
  result = ((w * (w + 1) / 2) * k) - n;
  if (result <= 0)
    cout << 0;
  else
    cout << result;
  return 0;
}
