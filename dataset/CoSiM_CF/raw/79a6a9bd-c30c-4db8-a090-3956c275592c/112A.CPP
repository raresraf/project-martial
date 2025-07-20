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

int compare(string a, string b) {
  for (int i = 0; i < a.length(); i++) {
    int letterA = tolower(a[i]);
    int letterB = tolower(b[i]);

    if (letterA < letterB)
      return -1;
    if (letterA > letterB)
      return 1;
  }

  return 0;
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string a, b;
  cin >> a >> b;

  cout << compare(a, b);

  return 0;
}
