
/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;

#define MOD 1000000007
#define _for(i, n) for (int i = 0; i < n; i++)
#define __for(i, n) for (int i = 1; i <= n; i++)
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define ALL(x) x.begin(), x.end()

bool isLucky(int a) {
  if (to_string(a).find("1") != string::npos ||
      to_string(a).find("2") != string::npos ||
      to_string(a).find("3") != string::npos ||
      to_string(a).find("5") != string::npos ||
      to_string(a).find("6") != string::npos ||
      to_string(a).find("8") != string::npos ||
      to_string(a).find("9") != string::npos ||
      to_string(a).find("0") != string::npos)
    return false;
  return true;
}

int main() {
  long long int n;
  cin >> n;
  int count = 0;
  string s = to_string(n);
  for (int i = 0; i < s.size(); i++) {
    if (s.at(i) == '4' || s.at(i) == '7')
      count++;
  }
  if (isLucky(count))
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
