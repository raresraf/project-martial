#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, num = 0;
  int coin[101];
  int total = 0;
  int sum = 0;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> coin[i];
    total += coin[i];
  }
  sort(coin, coin + n);
  for (int i = n - 1; i >= 0; i--) {
    sum += coin[i];
    num += 1;
    if (sum * 2 > total) {
      cout << num;
      return 0;
    }
  }
}
