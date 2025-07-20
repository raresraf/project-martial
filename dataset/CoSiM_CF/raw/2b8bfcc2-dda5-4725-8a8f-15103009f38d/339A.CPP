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

  string a;
  cin >> a;

  vector<char> list;

  for (int i = 0; i < a.length(); i++) {
    if (a[i] != '+') {
      list.push_back(a[i]);
    }
  }

  sort(list.begin(), list.end());

  for (int i = 0; i < list.size(); i++) {
    cout << list[i];

    if (i != list.size() - 1)
      cout << "+";
  }

  return 0;
}