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

  int n;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> n;
      if (n == 1) {
        cout << abs(i - 3) + abs(j - 3) << endl;
      }
    }
  }
  return 0;
}
