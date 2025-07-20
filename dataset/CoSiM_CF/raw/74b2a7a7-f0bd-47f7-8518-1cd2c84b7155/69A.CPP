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

int n;

int main() {

  scanf("%d", &n);
  int ans1 = 0, ans2 = 0, ans3 = 0;
  while (n--) {
    int x, y, z;
    scanf("%d%d%d", &x, &y, &z);
    ans1 += x, ans2 += y, ans3 += z;
  }

  printf("%s\n", (ans1 || ans2 || ans3) ? "NO" : "YES");

  return 0;
}
