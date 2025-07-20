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

  int num = 0;
  int posX = -1;
  int posY = 0;

  while (num != 1) {
    cin >> num;
    posX++;

    if (posX == 5) {
      posX = 0;
      posY++;
    }
  }

  int distance = abs(2 - posX) + abs(2 - posY);

  cout << distance;
  return 0;
}