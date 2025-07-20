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
  int letter = a[0];
  int count = 0;
  for (int i = 0; i < a.length() - 1; i++) {
    if (a[i] == a[i + 1])
      count++;
    else
      count = 0;

    if (count >= 6) {
      cout << "YES";
      return 0;
    }
  }

  cout << "NO";
  return 0;
}