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

  int num, value;
  cin >> num;
  int array[num];
  int sum = 0;
  for (int i = 0; i < num; i++) {
    cin >> value;
    array[i] = value;
    sum += value;
  }

  int n = sizeof(array) / sizeof(array[0]);

  sort(array, array + n, greater<int>());

  int take = 0;
  int numCoin = 0;
  double half = (double)sum / 2;
  while (take <= half) {
    take += array[numCoin];
    numCoin++;
  }

  cout << numCoin;

  return 0;
}