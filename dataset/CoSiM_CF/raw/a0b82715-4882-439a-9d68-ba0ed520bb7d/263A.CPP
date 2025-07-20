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

int grid[10][10];

int main() {

  fr(i, 1, 6) fr(j, 1, 6) {
    scanf("%d", &grid[i][j]);
    if (grid[i][j] == 1) {
      printf("%d\n", abs(i - 3) + abs(j - 3));
    }
  }

  return 0;
}
