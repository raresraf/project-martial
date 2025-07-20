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

  int n, t;
  string line;
  cin >> n >> t >> line;
  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (line[j] == 'B' && line[j + 1] == 'G') {
        line[j] = 'G';
        line[j + 1] = 'B';
        j++;
      }
    }
  }

  cout << line;

  return 0;
}