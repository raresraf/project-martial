#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  int k, n, w, ret = 0;
  cin >> k >> n >> w;
  for (int i = 0; i < w; i++)
    ret += (i + 1) * k;
  cout << max(0, ret - n) << endl;
  return 0;
}
