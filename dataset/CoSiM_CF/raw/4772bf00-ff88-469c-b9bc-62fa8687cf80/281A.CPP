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

string s;

int main() {

  cin >> s;

  s[0] += (islower(s[0]) ? ('A' - 'a') : 0);

  printf("%s\n", s.c_str());

  return 0;
}
