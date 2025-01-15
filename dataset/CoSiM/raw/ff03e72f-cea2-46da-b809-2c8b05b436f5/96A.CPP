#include <bits/stdc++.h>

using namespace std;

#define ms(ar, a) memset(ar, a, sizeof(ar))
#define fr(i, j, k) for (int(i) = (j); (i) < (k); (i)++)
#define pb push_back
#define mp make_pair
#define F first
#define S second

typedef long long ll;
typedef pair<int, int> pii;

const double PI = acos(-1.0);

template <class _T> inline string tostr(const _T &a) {
  ostringstream os("");
  os << a;
  return os.str();
}

// testing linux enviroment

string s;

int main() {

  cin >> s;

  int k = s.size();

  bool found = false;

  for (int i = 0; i < k - 7 + 1; i++) {
    string tmp = s.substr(i, 7);
    if (tmp == "0000000" || tmp == "1111111")
      found = true;
  }

  if (found)
    puts("YES");
  else
    puts("NO");

  return 0;
}
