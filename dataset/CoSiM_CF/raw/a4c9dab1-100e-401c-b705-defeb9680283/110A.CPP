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

  string num;
  cin >> num;
  int digit = 0;
  for (int i = 0; i < num.length(); i++) {
    if (num[i] == '4' || num[i] == '7')
      digit++;
  }
  if (digit == 4 || digit == 7)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}