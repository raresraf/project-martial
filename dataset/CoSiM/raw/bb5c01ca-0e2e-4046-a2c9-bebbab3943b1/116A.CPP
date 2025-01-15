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

  int numStop;
  cin >> numStop;
  int max = 0;
  int numPeople = 0;

  while (numStop != 0) {
    int on;
    int off;
    cin >> off >> on;

    numPeople += on;
    numPeople -= off;

    if (numPeople > max)
      max = numPeople;

    numStop--;
  }

  cout << max;

  return 0;
}