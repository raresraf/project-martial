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

  int num;
  cin >> num;
  int x, y, z;
  int totalX = 0;
  int totalY = 0;
  int totalZ = 0;
  while (num > 0) {
    cin >> x >> y >> z;
    totalX += x;
    totalY += y;
    totalZ += z;
    num--;
  }
  if (totalX == 0 && totalY == 0 && totalZ == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}