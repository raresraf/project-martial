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

int n, a[105];

bool cmp(int a, int b) { return a > b; }

int main() {

  int sum = 0;

  scanf("%d", &n);
  fr(i, 0, n) scanf("%d", &a[i]), sum += a[i];

  int tgt = (int)ceil((sum + 1) / 2.0);
  sort(a, a + n, cmp);
  sum = 0;
  int i = 0;

  while (sum < tgt)
    sum += a[i++];

  printf("%d\n", i);

  return 0;
}
