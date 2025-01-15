#include <bits/stdc++.h>
using namespace std;
#define fi(a) scanf("%d", &a);
#define fli(a) scanf("%ld", &a);
#define fll(a) scanf("%lld", &a);
#define pi(a) printf("%d\n", a);
#define ppi(i, a) printf("Case %d: %d\n", i, a);
#define ll long long
#define max 10000007

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  int n, k, a, cnt = 0, store;
  cin >> n >> k;
  for (int i = 1; i <= n; i++) {
    cin >> a;
    if (i == k) {
      store = a;
      if (a != 0)
        cnt++;
    } else if (i > k) {
      if (store == a && a != 0) {
        cnt++;
      }
    } else {
      if (a != 0)
        cnt++;
    }
  }
  cout << cnt << endl;
  return 0;
}
