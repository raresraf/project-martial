#pragma comment(linker, "/STACK:16777216")
#include <bits/stdc++.h>

using namespace std;

#define ms(ar, a) memset(ar, a, sizeof(ar))
#define fr(i, j, k) for (int(i) = (j); (i) < (k); (i)++)
#define db(x) cout << (#x) << " = " << x << endl;
#define pb push_back
#define mp make_pair
#define X first
#define Y second

typedef long long ll;
typedef pair<int, int> pii;

template <class _T> inline string tostr(const _T &a) {
  ostringstream os("");
  os << a;
  return os.str();
}

int n, k, a[55], t, zeros;

int main() {

  scanf("%d%d", &n, &k);
  fr(i, 0, n) scanf("%d", &a[i]), zeros += (a[i] == 0);
  t = a[k - 1]; // db(t);

  sort(a, a + n);
  int ans = lower_bound(a, a + n, t) - a;

  printf("%d\n", n - ans - (t == 0 ? zeros : 0));

  return 0;
}
