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

bool cnt[27];
string s;

int main() {

  cin >> s;
  int k = s.size(), val = 0;

  fr(i, 0, k) cnt[s[i] - 'a'] |= 1;
  fr(i, 0, 27) if (cnt[i]) val++;

  printf("%s\n", val & 1 ? "IGNORE HIM!" : "CHAT WITH HER!");

  return 0;
}
