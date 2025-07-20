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

  string a, b;
  int s1 = 0, s2 = 0;
  cin >> a >> b;
  for (int i = 0; i < a.size(); i++) {
    if (tolower(a[i]) < tolower(b[i])) {
      cout << -1 << endl;
      return 0;
    }
    if (tolower(a[i]) > tolower(b[i])) {
      cout << 1 << endl;
      return 0;
    }
  }
  cout << 0 << endl;
  return 0;
}
