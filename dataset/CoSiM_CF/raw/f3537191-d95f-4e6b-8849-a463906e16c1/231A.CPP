#pragma comment(linker, "/STACK:16777216")
#include <bits/stdc++.h>

using namespace std;

#define ms(ar, a) memset(ar, a, sizeof(ar))
#define fr(i, j, k) for (int(i) = (j); (i) < (k); (i)++)
#define db(x) cout << (#x) << " = " << x << endl;
#define pb push_back
#define mp make_pair
#define F X
#define S Y

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
  int resp = 0;

  while (n--) {
    int a, b, c;
    scanf("%d%d%d", &a, &b, &c);
    resp += ((a + b + c) > 1);
  }

  printf("%d\n", resp);

  return 0;
}
