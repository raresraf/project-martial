#include <bits/stdc++.h>
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;
int main() {
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL);
  int n, i;
  cin >> n;
  int a, b, count = 0, d = 0;
  loop(n) {
    cin >> a >> b;
    count += (b - a);
    d = max(d, count);
  }
  cout << d;
  cout << "\n";
  return 0;
}