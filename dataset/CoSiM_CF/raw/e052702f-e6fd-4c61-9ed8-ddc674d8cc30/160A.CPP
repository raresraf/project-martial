#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, i = 0, sum = 0, tem = 0;
  cin >> n;
  int coin[n];
  while (i < n) {
    cin >> coin[i];
    sum += coin[i];
    i++;
  }
  sort(coin + 0, coin + n, greater<int>());
  i = 0;
  while (tem <= sum / 2) {
    tem += coin[i];
    i++;
  }
  cout << i;
  return 0;
}
