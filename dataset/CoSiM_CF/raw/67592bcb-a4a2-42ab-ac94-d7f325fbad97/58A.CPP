/*
  Yadi Qian
*/

#include <bits/stdc++.h>

using namespace std;

const double pi = acos(-1.0);
typedef pair<int, int> PII;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<PII> VII;
typedef vector<VI> VVI;
typedef map<int, int> MPII;
typedef set<int> SETI;
typedef multiset<int> MSETI;
typedef long int int32;
typedef unsigned long int uint32;
typedef long long int int64;
typedef unsigned long long int uint64;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string s;
  cin >> s;

  vector<char> v = {'h', 'e', 'l', 'l', 'o'};
  int index = 0;

  for (int i = 0; i < s.length(); i++) {
    if (s[i] == v[index])
      index++;
  }

  if (index == 5)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}